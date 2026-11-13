#!/usr/bin/env python
import argparse, os, sys, math, time
import numpy as np
import openmm as mm
from openmm import app, unit
from pdbfixer import PDBFixer

KJ = unit.kilojoule_per_mole
KB = 0.008314462618

CHI = {
    'ARG': [('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','NE'),('CG','CD','NE','CZ')],
    'ASN': [('N','CA','CB','CG'),('CA','CB','CG','OD1')],
    'ASP': [('N','CA','CB','CG'),('CA','CB','CG','OD1')],
    'CYS': [('N','CA','CB','SG')],
    'GLN': [('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
    'GLU': [('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
    'HIS': [('N','CA','CB','CG'),('CA','CB','CG','ND1')],
    'ILE': [('N','CA','CB','CG1'),('CA','CB','CG1','CD1')],
    'LEU': [('N','CA','CB','CG'),('CA','CB','CG','CD1')],
    'LYS': [('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','CE'),('CG','CD','CE','NZ')],
    'MET': [('N','CA','CB','CG'),('CA','CB','CG','SD'),('CB','CG','SD','CE')],
    'PHE': [('N','CA','CB','CG'),('CA','CB','CG','CD1')],
    'SER': [('N','CA','CB','OG')],
    'THR': [('N','CA','CB','OG1')],
    'TRP': [('N','CA','CB','CG'),('CA','CB','CG','CD1')],
    'TYR': [('N','CA','CB','CG'),('CA','CB','CG','CD1')],
    'VAL': [('N','CA','CB','CG1')],
}

ALIAS = {'HID':'HIS','HIE':'HIS','HIP':'HIS','CYX':'CYS','ASH':'ASP',
         'GLH':'GLU','LYN':'LYS'}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--pocket-resids", required=True,
                    help="text file, one author residue number per line")
    ap.add_argument("--ref", default=None,
                    help="optional reference structure (e.g. stripped holo) to "
                         "track pocket RMSD against")
    ap.add_argument("--track-resid", type=int, default=None,
                    help="optional residue number for chi1 tracking")
    ap.add_argument("--steps", type=int, default=50000000)
    ap.add_argument("--delta-deg", type=float, default=30.0,
                    help="maximum local rotation magnitude in degrees")
    ap.add_argument("--jump-prob", type=float, default=0.3,
                    help="probability of a large rotamer-jump proposal")
    ap.add_argument("--jump-deg", type=float, default=120.0,
                    help="mean jump magnitude in degrees, with +/-20 degree jitter")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--md-every", type=int, default=2000,
                    help="run a short MD burst every N accepted moves (0=off)")
    ap.add_argument("--md-steps", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000,
                    help="write a trajectory frame every N attempted proposals")
    ap.add_argument("--ph", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restrain-backbone-k", type=float, default=1000.0,
                    help="harmonic restraint on backbone heavy atoms relative to the prepared starting receptor.")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--out-prefix", default="run")
    return ap.parse_args()


def prep(path, ph):
    fx = PDBFixer(filename=path)
    fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms()
    fx.removeHeterogens(keepWater=False)
    fx.addMissingHydrogens(ph)
    return fx.topology, fx.positions


def build_bond_graph(topology):
    adj = {i: set() for i in range(topology.getNumAtoms())}
    for a, b in topology.bonds():
        adj[a.index].add(b.index)
        adj[b.index].add(a.index)
    return adj


def moving_set(adj, ib, ic):
    seen = {ic}
    stack = [ic]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if {x, y} == {ib, ic}:
                continue
            if y == ib:
                return None
            if y not in seen:
                seen.add(y); stack.append(y)
    seen.discard(ic)
    return seen


def collect_chis(topology, pocket_resseq):
    adj = build_bond_graph(topology)
    chis = []
    for res in topology.residues():
        rid = _int(res.id)
        if rid not in pocket_resseq:
            continue
        base = ALIAS.get(res.name, res.name)
        if base not in CHI:
            continue
        name2idx = {a.name: a.index for a in res.atoms()}
        for quad in CHI[base]:
            if not all(n in name2idx for n in quad):
                continue
            ia, ib, ic, idd = (name2idx[n] for n in quad)
            mv = moving_set(adj, ib, ic)
            if mv is None or len(mv) == 0:
                continue
            chis.append((np.array(sorted(mv), dtype=int), ib, ic))
    return chis


def rotate(positions_nm, moving, ib, ic, angle_rad):
    p = positions_nm
    axis = p[ic] - p[ib]
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return positions_nm
    axis = axis / n
    origin = p[ib]
    v = p[moving] - origin
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    cross = np.cross(np.broadcast_to(axis, v.shape), v)
    dot = v @ axis
    v_rot = v * c + cross * s + np.outer(dot, axis) * (1 - c)
    out = positions_nm.copy()
    out[moving] = v_rot + origin
    return out


def pocket_ca_indices(topology, pocket_resseq):
    idx = []
    for res in topology.residues():
        if _int(res.id) in pocket_resseq:
            for a in res.atoms():
                if a.name == 'CA':
                    idx.append(a.index)
    return np.array(idx, dtype=int)


CHI1_NAMES = {
    'ARG':('N','CA','CB','CG'),'ASN':('N','CA','CB','CG'),'ASP':('N','CA','CB','CG'),
    'CYS':('N','CA','CB','SG'),'GLN':('N','CA','CB','CG'),'GLU':('N','CA','CB','CG'),
    'HIS':('N','CA','CB','CG'),'ILE':('N','CA','CB','CG1'),'LEU':('N','CA','CB','CG'),
    'LYS':('N','CA','CB','CG'),'MET':('N','CA','CB','CG'),'PHE':('N','CA','CB','CG'),
    'SER':('N','CA','CB','OG'),'THR':('N','CA','CB','OG1'),'TRP':('N','CA','CB','CG'),
    'TYR':('N','CA','CB','CG'),'VAL':('N','CA','CB','CG1'),
}


def chi1_indices(topology, resid):
    for res in topology.residues():
        if _int(res.id) != resid:
            continue
        base = ALIAS.get(res.name, res.name)
        if base not in CHI1_NAMES:
            return None
        name2idx = {a.name: a.index for a in res.atoms()}
        names = CHI1_NAMES[base]
        if not all(n in name2idx for n in names):
            return None
        return tuple(name2idx[n] for n in names)
    return None


def dihedral_deg(P, idx):
    p0, p1, p2, p3 = (P[i] for i in idx)
    b0, b1, b2 = p0-p1, p2-p1, p3-p2
    b1n = b1/np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n)*b1n
    w = b2 - np.dot(b2, b1n)*b1n
    return math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def which_well(chi):
    c = (chi + 360) % 360
    return "g+" if c < 120 else ("t" if c < 240 else "g-")


def rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    diff = Pc @ R.T - Qc
    return math.sqrt((diff ** 2).sum() / len(P))


def main():
    a = get_args()
    rng = np.random.default_rng(a.seed)
    beta = 1.0 / (KB * a.T)
    dmax = math.radians(a.delta_deg)

    pocket = set(int(x) for x in open(a.pocket_resids).read().split())
    print(f"pocket residues: {sorted(pocket)}")

    top, pos = prep(a.start, a.ph)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                             constraints=None, rigidWater=False,
                             removeCMMotion=False)

    if a.restrain_backbone_k > 0:
        ref = np.array(pos.value_in_unit(unit.nanometer))
        rest = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        rest.addGlobalParameter("k", a.restrain_backbone_k)
        for pn in ("x0", "y0", "z0"):
            rest.addPerParticleParameter(pn)
        bb = {"N", "CA", "C", "O"}
        n_r = 0
        for atom in top.atoms():
            if atom.name in bb:
                x0, y0, z0 = ref[atom.index]
                rest.addParticle(atom.index, [x0, y0, z0])
                n_r += 1
        system.addForce(rest)
        print(f"backbone restraint on {n_r} atoms, k="
              f"{a.restrain_backbone_k} kJ/mol/nm^2")
    integ = mm.LangevinMiddleIntegrator(a.T*unit.kelvin, 1/unit.picosecond,
                                        0.002*unit.picoseconds)
    plat = (mm.Platform.getPlatformByName(a.platform) if a.platform else None)
    sim = app.Simulation(top, system, integ, plat) if plat \
        else app.Simulation(top, system, integ)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=1000)

    chis = collect_chis(top, pocket)
    if not chis:
        sys.exit("no rotatable pocket chis found -- check --pocket-resids "
                 "numbering matches the PDB")
    print(f"{len(chis)} rotatable pocket chi angles")

    P = np.array(sim.context.getState(getPositions=True)
                 .getPositions().value_in_unit(unit.nanometer))
    E = sim.context.getState(getEnergy=True).getPotentialEnergy() \
        .value_in_unit(KJ)

    ca = pocket_ca_indices(top, pocket)
    ref_ca = None
    if a.ref:
        rtop, rpos = prep(a.ref, a.ph)
        rP = np.array([[v.x, v.y, v.z] for v in rpos])
        rca = pocket_ca_indices(rtop, pocket)
        if len(rca) == len(ca):
            ref_ca = rP[rca]
        else:
            print(f"[warn] ref pocket CA count {len(rca)} != {len(ca)}; "
                  f"RMSD tracking disabled")

    dcd = app.DCDFile(open(f"{a.out_prefix}.dcd", "wb"), top, 0.002)
    log = open(f"{a.out_prefix}.csv", "w")
    log.write("step,accepted,E_kJ,acc_rate,pocket_rmsd_A,chi1_deg\n")

    track_idx = None
    ref_chi1 = ref_well = None
    well_counts = {"g+": 0, "t": 0, "g-": 0}
    if a.track_resid is not None:
        track_idx = chi1_indices(top, a.track_resid)
        if track_idx is None:
            print(f"[warn] residue {a.track_resid} has no chi1; tracking off")
        else:
            if a.ref:
                rP_full = np.array([[v.x, v.y, v.z] for v in rpos])
                ridx = chi1_indices(rtop, a.track_resid)
                if ridx:
                    ref_chi1 = dihedral_deg(rP_full, ridx)
                    ref_well = which_well(ref_chi1)
                    print(f"reference chi1 of residue {a.track_resid} = "
                          f"{ref_chi1:+.0f} deg (well {ref_well})")
    start_chi1 = dihedral_deg(P, track_idx) if track_idx else None
    if start_chi1 is not None:
        print(f"start chi1 of residue {a.track_resid} = {start_chi1:+.0f} deg "
              f"(well {which_well(start_chi1)})")
    reached_ref_well = False
    prev_well = None
    n_transitions = 0

    best_E, best_P = E, P.copy()
    best_rmsd, best_rmsd_P = 1e9, None
    accepted = 0
    t0 = time.time()

    for step in range(1, a.steps + 1):
        mv, ib, ic = chis[rng.integers(len(chis))]
        if rng.random() < a.jump_prob:
            dtheta = math.radians(rng.choice([-1, 1]) *
                                  (a.jump_deg + rng.uniform(-20, 20)))
        else:
            dtheta = rng.uniform(-dmax, dmax)
        P_new = rotate(P, mv, ib, ic, dtheta)

        sim.context.setPositions(P_new * unit.nanometer)
        E_new = sim.context.getState(getEnergy=True).getPotentialEnergy() \
            .value_in_unit(KJ)
        dE = E_new - E
        if dE <= 0 or rng.random() < math.exp(-beta * dE):
            P, E = P_new, E_new
            accepted += 1

            if a.md_every and accepted % a.md_every == 0:
                sim.context.setVelocitiesToTemperature(a.T * unit.kelvin)
                sim.step(a.md_steps)
                st = sim.context.getState(getPositions=True, getEnergy=True)
                P = np.array(st.getPositions().value_in_unit(unit.nanometer))
                E = st.getPotentialEnergy().value_in_unit(KJ)
        else:
            sim.context.setPositions(P * unit.nanometer)  # revert

        r = np.nan
        if ref_ca is not None:
            r = rmsd(P[ca], ref_ca) * 10.0
            if r < best_rmsd:
                best_rmsd, best_rmsd_P = r, P.copy()
        if E < best_E:
            best_E, best_P = E, P.copy()

        chi1 = np.nan
        if track_idx is not None:
            chi1 = dihedral_deg(P, track_idx)
            w_now = which_well(chi1)
            well_counts[w_now] += 1
            if prev_well is not None and w_now != prev_well:
                n_transitions += 1
            prev_well = w_now
            if ref_well is not None and w_now == ref_well \
                    and which_well(start_chi1) != ref_well:
                reached_ref_well = True

        if step % a.save_every == 0:
            sim.context.setPositions(P * unit.nanometer)
            dcd.writeModel(sim.context.getState(getPositions=True)
                           .getPositions())
            log.write(f"{step},{accepted},{E:.2f},"
                      f"{accepted/step:.4f},{r if r==r else ''},"
                      f"{chi1 if chi1==chi1 else ''}\n")
            log.flush()
        if step % 5000 == 0:
            print(f"{step}/{a.steps} acc={accepted/step:.3f} E={E:.1f} "
                  f"rmsd={r:.2f}" if r == r else
                  f"{step}/{a.steps} acc={accepted/step:.3f} E={E:.1f}",
                  flush=True)

    log.close()
    for tag, Pw in [("best_energy", best_P),
                    ("closest_to_ref", best_rmsd_P)]:
        if Pw is None:
            continue
        with open(f"{a.out_prefix}_{tag}.pdb", "w") as fh:
            app.PDBFile.writeFile(top, Pw * unit.nanometer, fh)

    print(f"\ndone {a.steps} steps in {(time.time()-t0)/60:.1f} min")
    print(f"final acceptance = {accepted/a.steps:.3f}")
    print(f"best E = {best_E:.1f} kJ/mol")
    if ref_ca is not None:
        print(f"closest pocket RMSD to ref = {best_rmsd:.2f} A "
              f"-> {a.out_prefix}_closest_to_ref.pdb")
    if track_idx is not None:
        print(f"\nResidue {a.track_resid} chi1 observations:")
        for w in ("g+", "t", "g-"):
            print(f"  {w}: {well_counts[w]} observations")
        print(f"Observed well changes: {n_transitions}")

        if ref_well is not None:
            print(f"Reached reference well ({ref_well}): "
                  f"{'YES' if reached_ref_well else 'NO'}")



def _int(x):
    return int("".join(c for c in str(x) if c.isdigit() or c == '-') or -1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse, os, sys, time
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="apo PDB (same as MCMC)")
    ap.add_argument("--outdir", default="md_unrestrained")
    ap.add_argument("--ns", type=float, default=1000.0, help="production ns")
    ap.add_argument("--timestep", type=float, default=2.0, help="fs")
    ap.add_argument("--hmass", type=float, default=1.5, help="amu")
    ap.add_argument("--ph", type=float, default=7.0)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--cutoff", type=float, default=2.0,
                    help="nonbonded cutoff nm for implicit")
    ap.add_argument("--frame-ps", type=float, default=12.5,
                    help="MD trajectory frame interval in ps")
    ap.add_argument("--equil-ns", type=float, default=0.5)
    ap.add_argument("--platform", default="CUDA")
    return ap.parse_args()


def main():
    a = get_args()
    os.makedirs(a.outdir, exist_ok=True)
    sysxml = os.path.join(a.outdir, "system.xml")
    syspdb = os.path.join(a.outdir, "system.pdb")
    chk    = os.path.join(a.outdir, "checkpoint.chk")
    dcd    = os.path.join(a.outdir, "production.dcd")
    logf   = os.path.join(a.outdir, "md.log")
    timing = os.path.join(a.outdir, "timing.csv")

    dt = a.timestep * femtoseconds
    steps_per_ps = int(round(1000.0 / a.timestep))
    EQUIL_STEPS = int(a.equil_ns * 1000 * steps_per_ps)
    PROD_STEPS  = int(a.ns * 1000 * steps_per_ps)
    TARGET      = EQUIL_STEPS + PROD_STEPS
    REPORT      = int(round(a.frame_ps * steps_per_ps))

    if not (os.path.exists(sysxml) and os.path.exists(syspdb)):
        print(f"building implicit system from {a.start} ...", flush=True)
        fx = PDBFixer(filename=a.start)
        fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms()
        fx.removeHeterogens(keepWater=False)
        fx.addMissingHydrogens(a.ph)
        ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(fx.topology,
                                 nonbondedMethod=CutoffNonPeriodic,
                                 nonbondedCutoff=a.cutoff * nanometer,
                                 constraints=HBonds, rigidWater=False,
                                 hydrogenMass=a.hmass * amu,
                                 removeCMMotion=True)
        with open(sysxml, "w") as f:
            f.write(XmlSerializer.serialize(system))
        with open(syspdb, "w") as f:
            PDBFile.writeFile(fx.topology, fx.positions, f, keepIds=True)
        print(f"system: {fx.topology.getNumAtoms()} atoms (implicit)", flush=True)

    pdb = PDBFile(syspdb)
    with open(sysxml) as f:
        system = XmlSerializer.deserialize(f.read())
    integ = LangevinMiddleIntegrator(a.T * kelvin, 1.0 / picosecond, dt)
    plat = Platform.getPlatformByName(a.platform)
    props = {"Precision": "mixed"} if a.platform == "CUDA" else {}
    sim = Simulation(pdb.topology, system, integ, plat, props)
    print("platform:", sim.context.getPlatform().getName(), flush=True)

    if os.path.exists(chk):
        sim.loadCheckpoint(chk)
        print(f"resumed at step {sim.context.getStepCount()}", flush=True)
    else:
        sim.context.setPositions(pdb.positions)
        print("minimizing ...", flush=True)
        sim.minimizeEnergy(maxIterations=5000)
        sim.context.setVelocitiesToTemperature(a.T * kelvin)
        print(f"equilibrating {a.equil_ns} ns ...", flush=True)
        sim.step(EQUIL_STEPS)

    append = os.path.exists(dcd)
    sim.reporters.append(DCDReporter(dcd, REPORT, append=append))
    sim.reporters.append(StateDataReporter(
        open(logf, "a"), REPORT, step=True, time=True, potentialEnergy=True,
        temperature=True, speed=True))
    sim.reporters.append(StateDataReporter(
        sys.stdout, 50 * REPORT, step=True, time=True, temperature=True,
        speed=True, progress=True, remainingTime=True, totalSteps=TARGET))
    sim.reporters.append(CheckpointReporter(chk, 25 * REPORT))

    start_step = sim.context.getStepCount()
    remaining = TARGET - start_step
    print(f"target {TARGET} steps ({a.ns} ns prod, {a.frame_ps} ps/frame); "
          f"{remaining} remaining", flush=True)

    t0 = time.time()
    if remaining > 0:
        sim.step(remaining)
    wall = time.time() - t0
    done_step = sim.context.getStepCount()
    seg_ns = (done_step - start_step) * a.timestep / 1e6

    new = not os.path.exists(timing)
    with open(timing, "a") as fh:
        if new:
            fh.write("segment_ns,segment_wall_s,segment_gpu_h,ns_per_day,"
                     "cumulative_step\n")
        gpu_h = wall / 3600.0
        nsday = (seg_ns / gpu_h * 24.0) if gpu_h > 0 else float("nan")
        fh.write(f"{seg_ns:.3f},{wall:.1f},{gpu_h:.4f},{nsday:.1f},{done_step}\n")

    import pandas as pd
    tt = pd.read_csv(timing)
    tot_ns = tt.segment_ns.sum(); tot_h = tt.segment_gpu_h.sum()
    print(f"\nsegment: {seg_ns:.2f} ns in {wall/3600:.2f} GPU-h")
    print(f"CUMULATIVE: {tot_ns:.2f} ns in {tot_h:.2f} GPU-h "
          f"({tot_ns/tot_h if tot_h else float('nan'):.1f} ns/GPU-h, "
          f"{24*tot_ns/tot_h if tot_h else float('nan'):.0f} ns/day)")
    print(f"reached step {done_step}/{TARGET}")
    if done_step >= TARGET:
        print("DONE.")
    else:
        print("not finished -- resubmit to continue from checkpoint.")


if __name__ == "__main__":
    main()

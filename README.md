# Improving Molecular Docking Pose Recovery through Targeted Side-Chain Monte Carlo Sampling

This repository accompanies a research project conducted at the Research Science Institute.

This project addresses the issue of the accuracy of molecular docking being limited by protein flexibility. Since side-chain conformations in unbound (apo) structures may differ from those in ligand-bound (holo) structures, we investigate a targeted Markov Chain Monte Carlo (MCMC) approach that combines side-chain torsional perturbations with periodic molecular dynamics to generate an ensemble of receptor conformations for molecular docking.

The method was evaluated on protein-ligand complexes from the OpenBind EV-A71 2A protease benchmark, and achieved a 74.4% best-case ligand pose recovery rate within 2 Å RMSD, compared with 51.0% for the single apo structure and 59.5% for the restrained MD ensemble. These results demonstrate how targeted side-chain sampling can improve docking pose recovery. 

The simulations require an apo protein structure, with the MCMC script additionally requiring a file specifying the binding-pocket residues.

To generate a trajectory of MCMC, execute:
`python generate_MCMC.py --start apo.pdb --pocket-resids pocket_resids.txt`

The MCMC script outputs a trajectory and sampling diagnostics. The MD scripts output trajectories, topology files, and simulation logs.


To generate a trajectory of restrained MD, execute:
`python generate_restrained_md.py --start apo.pdb`


To generate a trajectory of unrestrained MD, execute:
`python generate_unrestrained_md.py --start apo.pdb`

These use the simulation parameters described in the accompanying paper. 


The generated trajectories were analyzed using pocket side-chain dihedral angles. Representative receptor conformations were selected using population-based and regular-space clustering. These conformations were then evaluated through molecular docking with SMINA, using ligand heavy-atom RMSD relative to experimental structures to measure pose recovery.

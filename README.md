# Improving Molecular Docking Pose Recovery through Targeted Side-Chain Monte Carlo Sampling

This repository accompanies a research project conducted at the Research Science Institute.

This project addresses the issue of the accuracy of molecular docking being limited by protein flexibility. Since side-chain conformations in unbound (apo) structures may differ from those in ligand-bound (holo) structures, we investigate a targeted Markov Chain Monte Carlo (MCMC) approach that combines side-chain torsional perturbations with periodic molecular dynamics to generate an ensemble of receptor conformations for molecular docking.

The method was evaluated on protein-ligand complexes from the OpenBind EV-A71 2A protease benchmark, and achieved a 74.4% best-case ligand pose recovery rate within 2 Å RMSD, compared with 51.0% for the single apo structure and 59.5% for the restrained MD ensemble. These results demonstrate how targeted side-chain sampling can improve docking pose recovery. 

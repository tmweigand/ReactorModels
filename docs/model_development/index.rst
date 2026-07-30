Model Development
========================

The domain of a fixed bed system contains fluid phase and solid phase. The fluid phase enters and leaves the domain through its 
boundaries (i.e. the top and bottom of a column) while the solid phase remains stationary within the domain. Fixed bed models are 
built upon the conservation of mass for a species in a phase over a differential column slice, consisting of terms for accumulation 
(first term) of species in a phase in the domain and advective transport (second term) of species in a phase through the domain

.. math::
    \pd{\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}}t

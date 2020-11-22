#include <Nyx.H>
#include <Hydro.H>
#include <constants_cosmo.H>

using namespace amrex;

using std::string;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
do_enforce_minimum_density(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real> const& state,
  AtomicRates* atomic_rates,
  const int NumSpec,
  amrex::Real const a_new,
  const amrex::Real gamma_minus_1,
  const amrex::Real small_dens,
  const amrex::Real small_temp)
{
    //Reset negative density to small_dens and zero out momenta:
    if(state(i,j,k,URHO) < small_dens)
    {
        for (int n = 0; n < NumSpec; ++n)
        {
            state(i,j,k,FirstSpec+n) *= small_dens / state(i,j,k,URHO);
        }

        state(i,j,k,URHO ) = small_dens;

        state(i,j,k,UMX) = 0.0;
        state(i,j,k,UMY) = 0.0;
        state(i,j,k,UMZ) = 0.0;

        amrex::Real dummy_pres = 1e200;
        amrex::Real eint_new = -1e200;
        amrex::Real Ne = 0.0;
        amrex::Real h_species = 0.76;

        // HACK HACK -- we can't yet do this call
        // Re-create "e" from {small_dens, small_temp}
        nyx_eos_given_RT(atomic_rates, gamma_minus_1, h_species, &eint_new, &dummy_pres, state(i,j,k,URHO),
						 small_temp, Ne,a_new);

        // Define (rho e) with small_dens and the new "e" 
        state(i,j,k,UEINT) = state(i,j,k,URHO) *  eint_new;

        // Here we effectively zero the KE so set (rho E) = (rho e)
        state(i,j,k,UEDEN) = state(i,j,k,UEINT);
    }
}

void
Nyx::enforce_minimum_density( MultiFab& S_old, MultiFab& S_new,
                              MultiFab& hydro_source,
                              amrex::Real dt, amrex::Real a_old, amrex::Real a_new)
{
    BL_PROFILE("Nyx::enforce_minimum_density()");

    if (verbose)
      amrex::Print() << "Enforce minimum density... " << std::endl;

    MultiFab::RegionTag amrhydro_tag("HydroUpdate_" + std::to_string(level));

    const amrex::Real a_half = 0.5 * (a_old + a_new);
    const amrex::Real a_half_inv = 1 / a_half;
    const amrex::Real a_oldsq = a_old * a_old;
    const amrex::Real a_newsq = a_new * a_new;
    const amrex::Real a_new_inv = 1.0 / a_new;
    const amrex::Real a_newsq_inv = 1.0 / a_newsq;
    const amrex::Real dt_a_new    = dt / a_new;

    int lnum_spec    = NumSpec;
    Real lsmall_dens = small_dens;
    Real lgamma_minus_1 = gamma - 1.0;
    Real lsmall_temp = small_temp;
    auto atomic_rates = atomic_rates_glob;

    // This set of dt should be used for Saxpy dt like setup
    for (amrex::MFIter mfi(S_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& uin = S_old.array(mfi);
        auto const& uout = S_new.array(mfi);
        auto const& hydro_src = hydro_source.array(mfi);

        //Unclear whether this should be part of previous ParallelFor
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
        {
          do_enforce_minimum_density(i, j, k, uout, atomic_rates, lnum_spec, a_new, lgamma_minus_1, lsmall_dens, lsmall_temp);
        });
    }
}

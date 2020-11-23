#include <Nyx.H>
#include <Hydro.H>
#include <constants_cosmo.H>

using namespace amrex;

using std::string;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_mu_for_enforce_min_density(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& mu_x,
  amrex::Array4<amrex::Real> const& mu_y,
  amrex::Array4<amrex::Real> const& mu_z,
  const amrex::Real small_dens)
{
    // NOTE: This is called on a tile box grown by one, and we have already done a FillBoundary/FillPatch.
    //       We include ghost cells in our tests for rho > small_dens so that we can create fluxes here
    //       regardless of which side of the grid boundary has the cell needing fixing.

    // NOTE: We trigger a correction when rho < small_dens, but if this is true, we adjust so that 
    //       rho == 1.1 * small_dens ... just to give ourselves some breathing room 
    Real target_density = 1.01 * small_dens;

    if (state(i,j,k,URHO) < small_dens)
    {
        // std::cout << "UPDATING " << IntVect(i,j,k) << " WITH INITIAL RHO " << state(i,j,k,URHO) << std::endl;
        // std::cout << " and x neighbors " <<  state(i+1,j,k,URHO) << " " <<  state(i-1,j,k,URHO) << std::endl;
        // std::cout << " and y neighbors " <<  state(i,j+1,k,URHO) << " " <<  state(i,j-1,k,URHO) << std::endl;
        // std::cout << " and z neighbors " <<  state(i,j,k+1,URHO) << " " <<  state(i,j,k-1,URHO) << std::endl;

        Real total_need  = target_density - state(i,j,k,URHO);

        // We only let each cell contribute 1/6 of its difference from target_density;
        //    this guarantees that the redistribution can't make a cell with (rho > target_density)
        //    go under target_density
        Real avail_from_ihi = amrex::max((state(i+1,j,k,URHO) - target_density) / 6.0, 0.0);
        Real avail_from_ilo = amrex::max((state(i-1,j,k,URHO) - target_density) / 6.0, 0.0);
        Real avail_from_jhi = amrex::max((state(i,j+1,k,URHO) - target_density) / 6.0, 0.0);
        Real avail_from_jlo = amrex::max((state(i,j-1,k,URHO) - target_density) / 6.0, 0.0);
        Real avail_from_khi = amrex::max((state(i,j,k+1,URHO) - target_density) / 6.0, 0.0);
        Real avail_from_klo = amrex::max((state(i,j,k-1,URHO) - target_density) / 6.0, 0.0);

        // std::cout << " avail x " << avail_from_ihi << " " << avail_from_ilo << std::endl;
        // std::cout << " avail y " << avail_from_jhi << " " << avail_from_jlo << std::endl;
        // std::cout << " avail z " << avail_from_khi << " " << avail_from_klo << std::endl;

        Real total_avail = avail_from_ihi + avail_from_ilo +
                           avail_from_jhi + avail_from_jlo +
                           avail_from_khi + avail_from_klo; 

        // std::cout << " total avail / total need " << total_avail << " " << total_need << std::endl;

        Real fac;
        if (total_need < total_avail)
            fac = total_need / total_avail;
        else
            fac = 1.0;

        Real from_ihi = fac * avail_from_ihi;
        Real from_ilo = fac * avail_from_ilo;
        Real from_jhi = fac * avail_from_jhi;
        Real from_jlo = fac * avail_from_jlo;
        Real from_khi = fac * avail_from_khi;
        Real from_klo = fac * avail_from_klo;

        if (from_ihi > 0)
            mu_x(i+1,j,k) =  from_ihi / (state(i+1,j,k,URHO) - state(i  ,j,k,URHO));
        if (from_ilo > 0)
            mu_x(i  ,j,k) = -from_ilo / (state(i  ,j,k,URHO) - state(i-1,j,k,URHO));
        if (from_jhi > 0)
            mu_y(i,j+1,k) =  from_jhi  / (state(i,j+1,k,URHO) - state(i,j  ,k,URHO));
        if (from_jlo > 0)
            mu_y(i,j  ,k) = -from_jlo  / (state(i,j  ,k,URHO) - state(i,j-1,k,URHO));
        if (from_khi > 0)
            mu_z(i,j,k+1) =  from_khi  / (state(i,j,k+1,URHO) - state(i,j,k  ,URHO));
        if (from_klo > 0)
            mu_z(i,j,k  ) = -from_klo  / (state(i,j,k  ,URHO) - state(i,j,k-1,URHO));
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
create_update_for_minimum_density(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& mu_x,
  amrex::Array4<amrex::Real> const& mu_y,
  amrex::Array4<amrex::Real> const& mu_z,
  amrex::Array4<amrex::Real> const& update,
  const int FirstSpec, const int NumSpec)
{
    for (int n = 0; n < state.nComp(); n++)
    {
        if (n == URHO or n == UMX or n == UMY or n == UMZ or n == UEDEN or n == UEINT) 
        {
            update(i,j,k,n) = mu_x(i+1,j,k) * (state(i+1,j,k,n) - state(i  ,j,k,n))
                             -mu_x(i  ,j,k) * (state(i  ,j,k,n) - state(i-1,j,k,n))
                             +mu_y(i,j+1,k) * (state(i,j+1,k,n) - state(i,j  ,k,n))
                             -mu_y(i,j  ,k) * (state(i,j  ,k,n) - state(i,j-1,k,n))
                             +mu_z(i,j,k+1) * (state(i,j,k+1,n) - state(i,j,k  ,n))
                             -mu_z(i,j,k  ) * (state(i,j,k  ,n) - state(i,j,k-1,n));
        } else {
            update(i,j,k,n) = 0.0;
        }
    }

    // Enforce that X is unchanged with change in rho
    for (int n = 0; n < NumSpec; ++n)
    {
        state(i,j,k,FirstSpec+n) *= (state(i,j,k,URHO) + update(i,j,k,URHO)) / state(i,j,k,URHO);
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

    int lnum_spec    = NumSpec;
    int lfirst_spec  = FirstSpec;
    Real lsmall_dens = small_dens;
  
    Real cur_time = state[State_Type].curTime();

    if (S_new.min(Density) < small_dens)
    {
        // We need to define this temporary because S_new only has one ghost cell and we need two.
        MultiFab Sborder;
        Sborder.define(grids, S_new.DistributionMap(), S_new.nComp(), 2);

        // Define face-based coefficients to be defined when enforcing minimum density 
        //     then used to enjoy the updates of all the other variables
        // The ghost face space is only needed as temp space; we only use "valid" faces...
        MultiFab mu_x(amrex::convert(grids,IntVect(1,0,0)), dmap, 1, 1);
        MultiFab mu_y(amrex::convert(grids,IntVect(0,1,0)), dmap, 1, 1);
        MultiFab mu_z(amrex::convert(grids,IntVect(0,0,1)), dmap, 1, 1);

        Real rho_old_min_before = S_old.min(0);
        Real rho_old_sum_before = S_old.sum(0);

        Real rho_new_min_before = S_new.min(0);
        Real rho_new_sum_before = S_new.sum(0);

        Real rho_new_min_after;
        Real rho_new_sum_after;

        Real rho_new_min = rho_new_min_before;

        bool too_low = (rho_new_min < small_dens);

        int iter = 0;

        // 10 is an arbitrary limit here -- just to make sure we don't get stuck here somehow 
        while (too_low and iter < 10)
        {
            // First make sure that all ghost cells are updated because we use them in defining fluxes
            FillPatch(*this, Sborder, 2, cur_time, State_Type, Density, Sborder.nComp());

            // Initialize to zero; these will only be non-zero at a face across which density is passed...
            mu_x.setVal(0.);
            mu_y.setVal(0.);
            mu_z.setVal(0.);

            // This will hold the update to each cell due to enforcing minimum density in a conservative way
            MultiFab update(grids , dmap, Sborder.nComp(), 0);
            update.setVal(0.);

            for (amrex::MFIter mfi(Sborder, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
            {
                 const amrex::Box& gbx = mfi.growntilebox(1);
                 auto const& sbord = Sborder.array(mfi);
                 auto const& mu_x_arr = mu_x.array(mfi);
                 auto const& mu_y_arr = mu_y.array(mfi);
                 auto const& mu_z_arr = mu_z.array(mfi);

                 //Unclear whether this should be part of previous ParallelFor
                 amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                 {
                   compute_mu_for_enforce_min_density(i, j, k, sbord, mu_x_arr, mu_y_arr, mu_z_arr, lsmall_dens);
                 });
            }

            for (amrex::MFIter mfi(Sborder, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
            {
                // Only update on valid cells
                const amrex::Box& bx = mfi.tilebox();
                auto const& sbord_arr = Sborder.array(mfi);
                auto const& mu_x_arr  = mu_x.array(mfi);
                auto const& mu_y_arr  = mu_y.array(mfi);
                auto const& mu_z_arr  = mu_z.array(mfi);
                auto const& upd_arr   = update.array(mfi);

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                {
                  create_update_for_minimum_density(i, j, k, sbord_arr, 
                                                    mu_x_arr, mu_y_arr, mu_z_arr, upd_arr, 
                                                    lfirst_spec, lnum_spec);
                });
            }

            S_new.plus(update,0,S_new.nComp(),0);

            if (S_new.contains_nan())
               amrex::Abort("NaN in enforce_minimum_density");

            rho_new_min_after = S_new.min(0);

            amrex::Print() << "After " << iter+1 << " iterations " << std::endl;
            amrex::Print() << "  MIN OF rho_old / rho_new / new rho_new " << 
                rho_old_min_before << " " << rho_new_min_before << " " << rho_new_min_after << std::endl;

            too_low = (rho_new_min_after < small_dens);
            iter++;

        } // iter

        rho_new_sum_after = S_new.sum(0);

        amrex::Print() << "After " << iter << " iterations " << std::endl;
        amrex::Print() << "  SUM OF rho_old / rho_new / new rho_new " << 
            rho_old_sum_before << " " << rho_new_sum_before << " " << rho_new_sum_after << std::endl;

        if (rho_new_min_after < small_dens)
           amrex::Abort("Not able to enforce small_dens this way after all");
    }
}

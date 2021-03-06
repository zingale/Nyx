#ifndef _Prob_H_
#define _Prob_H_

#include "Nyx.H"

enum Prob_Type_Index {
  z_in_comp = 0,
};

static void prob_param_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param, const amrex::Real initial_z)
{
    prob_param[z_in_comp] = initial_z;
}

static void prob_errtags_default(amrex::Vector<amrex::AMRErrorTag>& errtags)
{
    //Only include default tagging if NO_HYDRO=FALSE
#ifndef NO_HYDRO
    AMRErrorTagInfo info;
    errtags.push_back(AMRErrorTag(1,AMRErrorTag::GREATER,"overden",info));
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata_state(const int i,
                         const int j,
                         const int k,
                         amrex::Array4<amrex::Real> const& state,
                         amrex::GeometryData const& geomdata,
                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
{
  // This is the case where we have compiled with states defined 
  //  but they have only one component each so we fill them this way.
  if (state.nComp() == 1)
  {
    //Could be replaced with setVal
    state(i,j,k,0)    = 0.00;
  }
#ifndef NO_HYDRO
  // This is the regular case with NO_HYDRO = FALSE
  else if (state.nComp() > 1)
  {
    state(i,j,k,Density)    = 0.00; //1.5d0 * small_dens
    state(i,j,k,Xmom) = 0.00;
    state(i,j,k,Ymom) = 0.00;
    state(i,j,k,Zmom) = 0.00;

    // These will both be set later in the call to init_e.
    state(i,j,k,Eint) = 0.0;
    state(i,j,k,Eden) = 0.0;

    int FirstSpec = Eint + 1;   
    if (state.nComp() > FirstSpec)
    {
      //XHYDROGEN hardcoded to .76 init_managed
      state(i,j,k,FirstSpec  ) = .76;
      state(i,j,k,FirstSpec+1) = (1.0 - .76);
    }
  }
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata(const int i,
                   const int j,
                   const int k,
                   amrex::Array4<amrex::Real> const& state,
                   amrex::Array4<amrex::Real> const& diag_eos,
                   amrex::GeometryData const& geomdata,
                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
{

  prob_initdata_state(i, j ,k, state, geomdata, prob_param);

  // This is the case where we have compiled with states defined 
  //  but they have only one component each so we fill them this way.
  if (state.nComp() == 1 && diag_eos.nComp() == 1)
  {
    //Could be replaced with setVal
    diag_eos(i,j,k,0)    = 0.00;
  }
#ifndef NO_HYDRO
  // This is the regular case with NO_HYDRO = FALSE
  else if (state.nComp() > 1 && diag_eos.nComp() >= 2)
  {
    diag_eos(i,j,k,Temp_comp) = 0.0210*(1.00 + prob_param[z_in_comp])*(1.00 + prob_param[z_in_comp]);
    diag_eos(i,j,k,  Ne_comp) = 0.0;

    //Should be equivalent to inhomo_reion > 0 Nyx_setup.cpp
    if (diag_eos.nComp() > 2)
      diag_eos(i,j,k, Zhi_comp) = 7.5;
  }
#endif
}

#endif

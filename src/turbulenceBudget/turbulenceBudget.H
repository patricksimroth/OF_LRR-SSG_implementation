/*---------------------------------------------------------------------------*\
Class
    Foam::functionObjects::turbulenceBudget

Description
    Term-by-term budget of the mean momentum equation (every model) and of
    the transport equations of the compressible turbulence models in src/,
    evaluated on the current fields with the finite-volume operators and
    schemes of the solve.  Every term is
    kinematic (divided by rho) and carries the sign it has on the right-hand
    side of  0 = -D(phi)/Dt + sum(terms),  as in Pope (2000), Eq. (7.178): a
    positive value is a gain.  `convection` is that -D(phi)/Dt,
    -(div(rho U phi) - phi div(rho U))/rho with the div(phi,<field>) scheme.
    `imbalance` is the sum of all terms, ~0 for a converged steady solution
    except where a wall function fixes the cell value (omega in the first cell).

    Every model, fields budgetU:<term> (vector), the rhoSimpleFoam UEqn:
        U:     convection pressureGradient viscousStress reynoldsStress
               imbalance
        reynoldsStress is -div(rho R)/rho for a Reynolds-stress model (incl.
        the couplingFactor term), div(rho nut dev2(grad(U)^T))/rho
        + laplacian(rho nut, U)/rho for an eddy-viscosity model.  The
        nut-stabilisation pair of the Reynolds-stress models (explicit
        laplacian(rho nut, U) against the implicit one) cancels and is left out.

    SSGLRROmega2019, fields budget<R|omega>:<term> (R terms: symmTensor):
        R:     convection production pressureStrain dissipation
               viscousDiffusion turbulentDiffusion imbalance
        omega: convection production destruction crossDiffusion
               viscousDiffusion turbulentDiffusion imbalance

    SpalartAllmarasSALSA, fields budgetNuTilda:<term>:
               convection production destruction viscousDiffusion
               turbulentDiffusion (incl. the cb2 term) imbalance

    The terms repeat the expressions of the model sources (coefficients from
    the turbulenceProperties coeffs dictionary, defaults as there); a change
    of a model must be repeated here.  The imbalance checks the copy.

Usage
    Run with the solver in post-processing mode, before a `sets` function
    object that samples the fields:

        budget
        {
            type    turbulenceBudget;
            libs    ("<path>/libturbulenceBudget.so");
        }

    rhoSimpleFoam -postProcess -latestTime -dict <file>

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_turbulenceBudget_H
#define functionObjects_turbulenceBudget_H

#include "fvMeshFunctionObject.H"
#include "volFields.H"

namespace Foam
{
namespace functionObjects
{

class turbulenceBudget
:
    public fvMeshFunctionObject
{
    // Private Member Functions

        template<class Type>
        void storeTerm
        (
            const word& equation,
            const word& term,
            const tmp<GeometricField<Type, fvPatchField, volMesh>>& tfld
        );

        void budgetMomentum();

        void budgetSSGLRROmega2019();

        void budgetSpalartAllmarasSALSA();


public:

    TypeName("turbulenceBudget");


    // Constructors

        turbulenceBudget
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    virtual ~turbulenceBudget() = default;


    // Member Functions

        virtual bool read(const dictionary& dict);

        virtual bool execute();

        virtual bool write();
};


} // End namespace functionObjects
} // End namespace Foam

#endif

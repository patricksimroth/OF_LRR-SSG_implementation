#include "SSGLRROmega2019.H"
#include "fvOptions.H"
#include "bound.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
void SSGLRROmega2019<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = k_/omega_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SSGLRROmega2019<BasicTurbulenceModel>::F1() const
{
    const volScalarField gradProduct
    (
        max
        (
            fvc::grad(k_) & fvc::grad(omega_),
            dimensionedScalar(dimless/pow3(dimTime), 0)
        )
    );
    const volScalarField CD
    (
        max
        (
            1.712*gradProduct/omega_,
            dimensionedScalar(sqr(dimless/dimTime), SMALL)
        )
    );
    const volScalarField zeta
    (
        min
        (
            max
            (
                sqrt(k_)/(Cmu_*omega_*y_),
                500*this->nu()/(omega_*sqr(y_))
            ),
            4*0.856*k_/(CD*sqr(y_))
        )
    );
    return tanh(pow4(zeta));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SSGLRROmega2019<BasicTurbulenceModel>::lsc() const
{
    const volScalarField ratio
    (
        sqrt(k_)/(pow(Cmu_, 0.25)*kappa_*omega_*y_)
    );
    const volScalarField chi(max((ratio - scalar(1))*sqr(ratio), scalar(0)));
    return 0.5*(scalar(1) + tanh(lscA_*(chi - lscChiT_)));
}


template<class BasicTurbulenceModel>
SSGLRROmega2019<BasicTurbulenceModel>::SSGLRROmega2019
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    ReynoldsStress<RASModel<BasicTurbulenceModel>>
    (
        type, alpha, rho, U, alphaRhoPhi, phi, transport, propertiesName
    ),
    Cmu_(dimensioned<scalar>::getOrAddToDict("Cmu", this->coeffDict_, 0.09)),
    kappa_(dimensioned<scalar>::getOrAddToDict("kappa", this->coeffDict_, 0.41)),
    lscA_(dimensioned<scalar>::getOrAddToDict("lscA", this->coeffDict_, 31)),
    lscChiT_(dimensioned<scalar>::getOrAddToDict("lscChiT", this->coeffDict_, 1)),
    k_
    (
        IOobject
        (
            "k", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        0.5*tr(this->R_)
    ),
    omega_
    (
        IOobject
        (
            "omega", this->runTime_.timeName(), this->mesh_,
            IOobject::MUST_READ, IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    y_(wallDist::New(this->mesh_).y())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
        this->boundNormalStress(this->R_);
        bound(omega_, this->omegaMin_);
        k_ = 0.5*tr(this->R_);
    }
}


template<class BasicTurbulenceModel>
bool SSGLRROmega2019<BasicTurbulenceModel>::read()
{
    if (ReynoldsStress<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        kappa_.readIfPresent(this->coeffDict());
        lscA_.readIfPresent(this->coeffDict());
        lscChiT_.readIfPresent(this->coeffDict());
        return true;
    }
    return false;
}


template<class BasicTurbulenceModel>
void SSGLRROmega2019<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_) return;

    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    volSymmTensorField& R = this->R_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    ReynoldsStress<RASModel<BasicTurbulenceModel>>::correct();

    const volTensorField gradU(fvc::grad(U));
    const volSymmTensorField P(-twoSymm(R & gradU));
    const volScalarField G(this->GName(), 0.5*mag(tr(P)));
    const volScalarField blend(F1());

    const volScalarField alphaOmega(0.44 + blend*(0.5556 - 0.44));
    const volScalarField betaOmega(0.0828 + blend*(0.075 - 0.0828));
    const volScalarField sigmaOmega(0.856 + blend*(0.5 - 0.856));
    const volScalarField sigmaD(1.712*(scalar(1) - blend));

    omega_.boundaryFieldRef().updateCoeffs();
    omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();

    const volScalarField crossDiff
    (
        sigmaD/omega_*max
        (
            fvc::grad(k_) & fvc::grad(omega_),
            dimensionedScalar(dimless/pow3(dimTime), 0)
        )
    );
    const volScalarField FLsc(lsc());

    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*(this->nu() + sigmaOmega*k_/omega_), omega_)
     ==
        alpha*rho*alphaOmega*omega_/k_*G
      - fvm::Sp(alpha*rho*(scalar(1) - FLsc)*betaOmega*omega_, omega_)
      + alpha*rho*crossDiff
      + fvOptions(alpha, rho, omega_)
    );
    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);

    const volScalarField epsilon(Cmu_*k_*omega_);
    const volSymmTensorField a(dev(R)/k_);
    const volSymmTensorField S(symm(gradU));
    const volTensorField W(skew(gradU));

    const volScalarField C1(1.7 + blend*(1.8 - 1.7));
    const volScalarField C1s(0.9*(scalar(1) - blend));
    const volScalarField C2(1.05*(scalar(1) - blend));
    const volScalarField C3s(0.65*(scalar(1) - blend));
    const volScalarField C4(0.625 + blend*((9*0.52 + 6.0)/11.0 - 0.625));
    const volScalarField C5(0.2 + blend*((-7*0.52 + 10.0)/11.0 - 0.2));
    const volScalarField A(C1*epsilon + C1s*G);
    const volScalarField D(0.5*Cmu_*blend + (2.0/3.0)*0.22*(scalar(1)-blend));

    tmp<fvSymmTensorMatrix> REqn
    (
        fvm::ddt(alpha, rho, R)
      + fvm::div(alphaRhoPhi, R)
      - fvm::laplacian(alpha*rho*(this->nu() + D/Cmu_*k_/omega_), R)
      + fvm::Sp(alpha*rho*A/k_, R)
     ==
        alpha*rho
       *(
            P + (2.0/3.0)*A*I - (2.0/3.0)*epsilon*I
          + C2*epsilon*dev(innerSqr(a))
          + k_*((0.8 - C3s*mag(a))*dev(S)
          + C4*devTwoSymm(a&S) + C5*twoSymm(a&W))
        )
      + fvOptions(alpha, rho, R)
    );
    REqn.ref().relax();
    fvOptions.constrain(REqn.ref());
    solve(REqn);
    fvOptions.correct(R);

    this->boundNormalStress(R);
    k_ = 0.5*tr(R);
    bound(k_, this->kMin_);
    correctNut();
    this->correctWallShearStress(R);
}

} // namespace RASModels
} // namespace Foam

#ifndef SSGLRROmega2019_H
#define SSGLRROmega2019_H

#include "RASModel.H"
#include "ReynoldsStress.H"
#include "wallDist.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
class SSGLRROmega2019
:
    public ReynoldsStress<RASModel<BasicTurbulenceModel>>
{
    SSGLRROmega2019(const SSGLRROmega2019&) = delete;
    void operator=(const SSGLRROmega2019&) = delete;

protected:
    dimensionedScalar Cmu_;
    dimensionedScalar kappa_;
    dimensionedScalar lscA_;
    dimensionedScalar lscChiT_;

    volScalarField k_;
    volScalarField omega_;
    const volScalarField& y_;

    tmp<volScalarField> F1() const;
    tmp<volScalarField> lsc() const;
    virtual void correctNut();

public:
    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    TypeName("SSGLRROmega2019");

    SSGLRROmega2019
    (
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName
    );

    virtual ~SSGLRROmega2019() = default;
    virtual bool read();
    virtual tmp<volScalarField> k() const { return k_; }
    virtual tmp<volScalarField> epsilon() const { return Cmu_*k_*omega_; }
    virtual void correct();
};

} // namespace RASModels
} // namespace Foam

#ifdef NoRepository
    #include "SSGLRROmega2019.C"
#endif

#endif

#include "turbulenceBudget.H"
#include "turbulentFluidThermoModel.H"
#include "wallDist.H"
#include "fvc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(turbulenceBudget, 0);
    addToRunTimeSelectionTable(functionObject, turbulenceBudget, dictionary);
}
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class Type>
void Foam::functionObjects::turbulenceBudget::storeTerm
(
    const word& equation,
    const word& term,
    const tmp<GeometricField<Type, fvPatchField, volMesh>>& tfld
)
{
    word fieldName("budget" + equation + ':' + term);
    store(fieldName, tfld);
}


void Foam::functionObjects::turbulenceBudget::budgetMomentum()
{
    const auto& turb = lookupObject<compressible::turbulenceModel>
    (
        turbulenceModel::propertiesName
    );

    const volScalarField& rho = turb.rho();
    const surfaceScalarField& phi = turb.alphaRhoPhi();
    const volVectorField& U = turb.U();
    const volScalarField& p = lookupObject<volScalarField>("p");
    const tmp<volScalarField> tnu(turb.nu());
    const volScalarField& nu = tnu();
    const tmp<volScalarField> tnut(turb.nut());
    const volScalarField& nut = tnut();

    const volTensorField gradU(fvc::grad(U));
    const volTensorField devGradUT(dev2(T(gradU)));

    tmp<volVectorField> conv(-fvc::div(phi, U, "div(phi,U)")/rho);
    tmp<volVectorField> gradP(-fvc::grad(p, "grad(p)")/rho);
    tmp<volVectorField> viscous
    (
        (
            fvc::laplacian(rho*nu, U)
          + fvc::div
            (
                rho*nu*devGradUT,
                "div(((rho*(thermo:mu|rho))*dev2(T(grad(U)))))"
            )
        )/rho
    );

    tmp<volVectorField> reynolds;
    if (foundObject<volSymmTensorField>("R"))
    {
        const volSymmTensorField& R = lookupObject<volSymmTensorField>("R");
        const scalar couplingFactor =
            turb.subDict("RAS").optionalSubDict(turb.type() + "Coeffs")
           .getOrDefault<scalar>("couplingFactor", 0);

        reynolds = -fvc::div(rho*R, "div((rho*R))")/rho;
        if (couplingFactor > 0)
        {
            reynolds.ref() -=
                couplingFactor
               *(
                    fvc::div(rho*nut*gradU, "div(devRhoReff)")
                  - fvc::laplacian(rho*nut, U, "laplacian(nuEff,U)")
                )/rho;
        }
    }
    else
    {
        reynolds =
        (
            fvc::laplacian(rho*nut, U)
          + fvc::div
            (
                rho*nut*devGradUT,
                "div(((rho*nuEff)*dev2(T(grad(U)))))"
            )
        )/rho;
    }

    tmp<volVectorField> imbalance
    (
        conv() + gradP() + viscous() + reynolds()
    );

    storeTerm("U", "convection", conv);
    storeTerm("U", "pressureGradient", gradP);
    storeTerm("U", "viscousStress", viscous);
    storeTerm("U", "reynoldsStress", reynolds);
    storeTerm("U", "imbalance", imbalance);
}


void Foam::functionObjects::turbulenceBudget::budgetSSGLRROmega2019()
{
    const auto& turb = lookupObject<compressible::turbulenceModel>
    (
        turbulenceModel::propertiesName
    );
    const dictionary coeffs
    (
        turb.subDict("RAS").optionalSubDict("SSGLRROmega2019Coeffs")
    );
    const scalar Cmu = coeffs.getOrDefault<scalar>("Cmu", 0.09);
    const scalar kappa = coeffs.getOrDefault<scalar>("kappa", 0.41);
    const scalar lscA = coeffs.getOrDefault<scalar>("lscA", 31);
    const scalar lscChiT = coeffs.getOrDefault<scalar>("lscChiT", 1);

    const volScalarField& rho = turb.rho();
    const surfaceScalarField& phi = turb.alphaRhoPhi();
    const volVectorField& U = turb.U();
    const volSymmTensorField& R = lookupObject<volSymmTensorField>("R");
    const volScalarField& omega = lookupObject<volScalarField>("omega");
    const tmp<volScalarField> tk(turb.k());
    const volScalarField& k = tk();
    const tmp<volScalarField> tnu(turb.nu());
    const volScalarField& nu = tnu();
    const volScalarField& y = wallDist::New(mesh_).y();

    const volTensorField gradU(fvc::grad(U));
    const volSymmTensorField P(-twoSymm(R & gradU));
    const volScalarField G(0.5*mag(tr(P)));

    // F1(), as SSGLRROmega2019::F1
    const volScalarField gradKgradOmega
    (
        max
        (
            fvc::grad(k) & fvc::grad(omega),
            dimensionedScalar(dimless/pow3(dimTime), 0)
        )
    );
    const volScalarField CD
    (
        max
        (
            1.712*gradKgradOmega/omega,
            dimensionedScalar(sqr(dimless/dimTime), SMALL)
        )
    );
    const volScalarField zeta
    (
        min
        (
            max
            (
                sqrt(k)/(Cmu*omega*y),
                500*nu/(omega*sqr(y))
            ),
            4*0.856*k/(CD*sqr(y))
        )
    );
    const volScalarField blend(tanh(pow4(zeta)));

    // lsc(), as SSGLRROmega2019::lsc
    const volScalarField ratio(sqrt(k)/(pow(Cmu, 0.25)*kappa*omega*y));
    const volScalarField chi(max((ratio - scalar(1))*sqr(ratio), scalar(0)));
    const volScalarField FLsc(0.5*(scalar(1) + tanh(lscA*(chi - lscChiT))));

    // omega equation
    {
        const volScalarField alphaOmega(0.44 + blend*(0.5556 - 0.44));
        const volScalarField betaOmega(0.0828 + blend*(0.075 - 0.0828));
        const volScalarField sigmaOmega(0.856 + blend*(0.5 - 0.856));
        const volScalarField sigmaD(1.712*(scalar(1) - blend));

        tmp<volScalarField> conv
        (
            -fvc::div(phi, omega, "div(phi,omega)")/rho
        );
        tmp<volScalarField> prod(alphaOmega*omega/k*G);
        tmp<volScalarField> destr(-(scalar(1) - FLsc)*betaOmega*sqr(omega));
        tmp<volScalarField> cross(sigmaD/omega*gradKgradOmega);
        tmp<volScalarField> viscDiff(fvc::laplacian(rho*nu, omega)/rho);
        tmp<volScalarField> turbDiff
        (
            fvc::laplacian(rho*sigmaOmega*k/omega, omega)/rho
        );
        tmp<volScalarField> imbalance
        (
            conv() + prod() + destr() + cross() + viscDiff() + turbDiff()
        );

        storeTerm("Omega", "convection", conv);
        storeTerm("Omega", "production", prod);
        storeTerm("Omega", "destruction", destr);
        storeTerm("Omega", "crossDiffusion", cross);
        storeTerm("Omega", "viscousDiffusion", viscDiff);
        storeTerm("Omega", "turbulentDiffusion", turbDiff);
        storeTerm("Omega", "imbalance", imbalance);
    }

    // R equation
    {
        const volScalarField epsilon(Cmu*k*omega);
        const volSymmTensorField a(dev(R)/k);
        const volSymmTensorField S(symm(gradU));
        const volTensorField W(skew(gradU));

        const volScalarField C1(1.7 + blend*(1.8 - 1.7));
        const volScalarField C1s(0.9*(scalar(1) - blend));
        const volScalarField C2(1.05*(scalar(1) - blend));
        const volScalarField C3s(0.65*(scalar(1) - blend));
        const volScalarField C4(0.625 + blend*((9*0.52 + 6.0)/11.0 - 0.625));
        const volScalarField C5(0.2 + blend*((-7*0.52 + 10.0)/11.0 - 0.2));
        const volScalarField A(C1*epsilon + C1s*G);
        const volScalarField D
        (
            0.5*Cmu*blend + (2.0/3.0)*0.22*(scalar(1) - blend)
        );
        const dimensionedSymmTensor Id("I", dimless, symmTensor::I);

        tmp<volSymmTensorField> conv(-fvc::div(phi, R, "div(phi,R)")/rho);
        tmp<volSymmTensorField> prod(new volSymmTensorField(P));
        // The implicit -A/k R of the solve plus the explicit 2/3 A I and
        // the C2, C3*, C4, C5 terms: the whole SSG/LRR pressure-strain model.
        tmp<volSymmTensorField> pressureStrain
        (
            (2.0/3.0)*A*Id - A/k*R
          + C2*epsilon*dev(innerSqr(a))
          + k*((0.8 - C3s*mag(a))*dev(S)
          + C4*devTwoSymm(a & S) + C5*twoSymm(a & W))
        );
        tmp<volSymmTensorField> diss(-(2.0/3.0)*epsilon*Id);
        tmp<volSymmTensorField> viscDiff(fvc::laplacian(rho*nu, R)/rho);
        tmp<volSymmTensorField> turbDiff
        (
            fvc::laplacian(rho*D/Cmu*k/omega, R)/rho
        );
        tmp<volSymmTensorField> imbalance
        (
            conv() + prod() + pressureStrain() + diss() + viscDiff()
          + turbDiff()
        );

        storeTerm("R", "convection", conv);
        storeTerm("R", "production", prod);
        storeTerm("R", "pressureStrain", pressureStrain);
        storeTerm("R", "dissipation", diss);
        storeTerm("R", "viscousDiffusion", viscDiff);
        storeTerm("R", "turbulentDiffusion", turbDiff);
        storeTerm("R", "imbalance", imbalance);
    }
}


void Foam::functionObjects::turbulenceBudget::budgetSpalartAllmarasSALSA()
{
    const auto& turb = lookupObject<compressible::turbulenceModel>
    (
        turbulenceModel::propertiesName
    );
    const dictionary coeffs
    (
        turb.subDict("RAS").optionalSubDict("SpalartAllmarasSALSACoeffs")
    );
    const scalar sigmaNut = coeffs.getOrDefault<scalar>("sigmaNut", 0.66666);
    const scalar kappa = coeffs.getOrDefault<scalar>("kappa", 0.41);
    const scalar Cb1 = coeffs.getOrDefault<scalar>("Cb1", 0.1355);
    const scalar Cb2 = coeffs.getOrDefault<scalar>("Cb2", 0.0);
    const scalar Cw2 = coeffs.getOrDefault<scalar>("Cw2", 0.3);
    const scalar Cw3 = coeffs.getOrDefault<scalar>("Cw3", 2.0);
    const scalar Cv1 = coeffs.getOrDefault<scalar>("Cv1", 7.1);
    const scalar Cs = coeffs.getOrDefault<scalar>("Cs", 0.3);
    const bool ft2On = coeffs.getOrDefault<Switch>("ft2", false);
    const scalar Ct3 = coeffs.getOrDefault<scalar>("Ct3", 1.2);
    const scalar Ct4 = coeffs.getOrDefault<scalar>("Ct4", 0.5);
    const dimensionedScalar rhoInf
    (
        dimDensity, coeffs.getOrDefault<scalar>("rhoInf", 1.0)
    );
    const bool rMod = coeffs.getOrDefault<Switch>("useRmod", false);
    const bool sMod = coeffs.getOrDefault<Switch>("useSmod", true);

    const volScalarField& rho = turb.rho();
    const surfaceScalarField& phi = turb.alphaRhoPhi();
    const volVectorField& U = turb.U();
    const volScalarField& nuTilda = lookupObject<volScalarField>("nuTilda");
    const tmp<volScalarField> tnu(turb.nu());
    const volScalarField& nu = tnu();
    const volScalarField& d = wallDist::New(mesh_).y();

    const volScalarField chi(nuTilda/nu);
    const volScalarField chi3(pow3(chi));
    const volScalarField fv1(chi3/(chi3 + pow3(Cv1)));
    volScalarField ft2(0*chi);
    if (ft2On)
    {
        ft2 = Ct3*exp(-Ct4*sqr(chi));
    }
    const volTensorField gradU(fvc::grad(U));
    const volScalarField Omega(sqrt(2.0)*mag(skew(gradU)));
    const volScalarField sqrKappaD(sqr(kappa*d));
    const dimensionedScalar eps(dimless/dimTime, SMALL);
    const dimensionedScalar epsChi(dimless, SMALL);

    // S as in SpalartAllmarasSALSABase::correct: Sstar (useSmod) or Stilda
    volScalarField S(sqrt(2.0)*mag(dev(symm(gradU))));
    if (!sMod)
    {
        const volScalarField fv2(scalar(1) - chi/(scalar(1) + chi*fv1));
        S = max(Omega + fv2*nuTilda/sqrKappaD, Cs*Omega);
    }

    // GammaEff
    const volScalarField rGamma
    (
        min(nuTilda/(max(S, eps)*sqrKappaD), scalar(10))
    );
    const volScalarField gamma
    (
        max
        (
            pow(scalar(1.01)*rGamma, scalar(0.65)),
            pow(max(scalar(0), scalar(1) - tanh(chi/scalar(68))), scalar(0.65))
        )
    );
    const volScalarField Cb1Eff
    (
        Cb1*sqrt(min(scalar(1.25), max(gamma, scalar(0.75))))
    );
    const volScalarField Cw1Eff
    (
        Cb1Eff/sqr(kappa) + (scalar(1) + Cb2)/sigmaNut
    );

    volScalarField Shat(S);
    if (sMod)
    {
        Shat = S*(1/max(chi, epsChi) + fv1);
    }

    // fw (r) or fwMod (rMod) on S, or on Omega for rMod without sMod
    volScalarField r(min(nuTilda/(max(S, eps)*sqrKappaD), scalar(10)));
    if (rMod)
    {
        const volScalarField& Sb = sMod ? S : Omega;
        const volScalarField StildaTmp(Sb*(1/max(chi, epsChi) + fv1));
        const volScalarField Psi(sqrt(rhoInf/rho)*nuTilda/sqrKappaD);
        r = min(1.6*tanh(0.7*Psi/max(StildaTmp, eps)), scalar(10));
    }
    const volScalarField g(r + Cw2*(pow6(r) - r));
    const volScalarField fw
    (
        g*pow((1 + pow6(Cw3))/(pow6(g) + pow6(Cw3)), 1.0/6.0)
    );

    tmp<volScalarField> conv
    (
        -fvc::div(phi, nuTilda, "div(phi,nuTilda)")/rho
    );
    tmp<volScalarField> prod(Cb1Eff*Shat*nuTilda*(scalar(1) - ft2));
    tmp<volScalarField> destr
    (
        -(Cw1Eff*fw - Cb1Eff/sqr(kappa)*ft2)*nuTilda*nuTilda/sqr(d)
    );
    tmp<volScalarField> viscDiff
    (
        fvc::laplacian(rho*nu/sigmaNut, nuTilda)/rho
    );
    tmp<volScalarField> turbDiff
    (
        fvc::laplacian(rho*nuTilda/sigmaNut, nuTilda)/rho
      + Cb2/sigmaNut*magSqr(fvc::grad(nuTilda))
    );
    tmp<volScalarField> imbalance
    (
        conv() + prod() + destr() + viscDiff() + turbDiff()
    );

    storeTerm("NuTilda", "convection", conv);
    storeTerm("NuTilda", "production", prod);
    storeTerm("NuTilda", "destruction", destr);
    storeTerm("NuTilda", "viscousDiffusion", viscDiff);
    storeTerm("NuTilda", "turbulentDiffusion", turbDiff);
    storeTerm("NuTilda", "imbalance", imbalance);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::turbulenceBudget::turbulenceBudget
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::turbulenceBudget::read(const dictionary& dict)
{
    return fvMeshFunctionObject::read(dict);
}


bool Foam::functionObjects::turbulenceBudget::execute()
{
    if (!foundObject<compressible::turbulenceModel>(turbulenceModel::propertiesName))
    {
        WarningInFunction
            << "No compressible turbulence model - no budget" << endl;
        return false;
    }
    const word model
    (
        lookupObject<compressible::turbulenceModel>
        (
            turbulenceModel::propertiesName
        ).type()
    );

    budgetMomentum();

    if (model == "SSGLRROmega2019")
    {
        budgetSSGLRROmega2019();
    }
    else if (model == "SpalartAllmarasSALSA")
    {
        budgetSpalartAllmarasSALSA();
    }
    else
    {
        Info<< type() << " " << name() << ": no turbulence budget for "
            << model << ", momentum only" << endl;
    }

    Info<< type() << " " << name() << ": " << model << " budget stored" << nl
        << endl;

    return true;
}


bool Foam::functionObjects::turbulenceBudget::write()
{
    return true;
}


// ************************************************************************* //

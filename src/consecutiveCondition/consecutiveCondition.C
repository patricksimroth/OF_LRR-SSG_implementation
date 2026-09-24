/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

\*---------------------------------------------------------------------------*/

#include "consecutiveCondition.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
namespace runTimeControls
{
    defineTypeNameAndDebug(consecutiveCondition, 0);
    addToRunTimeSelectionTable
    (
        runTimeCondition,
        consecutiveCondition,
        dictionary
    );
}
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::runTimeControls::consecutiveCondition::
consecutiveCondition
(
    const word& name,
    const objectRegistry& obr,
    const dictionary& dict,
    stateFunctionObject& state
)
:
    runTimeCondition(name, obr, dict, state),
    nIter_(dict.get<label>("nIter")),
    count_(0),
    condition_
    (
        runTimeCondition::New(name, obr, dict.subDict("condition"), state)
    )
{
    if (nIter_ < 1)
    {
        FatalIOErrorInFunction(dict)
            << "nIter must be >= 1, found " << nIter_
            << exit(FatalIOError);
    }

    this->conditionDict().readIfPresent("consecutiveCount", count_);
}


// * * * * * * * * * * * * * * Public Member Functions * * * * * * * * * * * //

bool Foam::functionObjects::runTimeControls::consecutiveCondition::apply()
{
    if (!active_)
    {
        return true;
    }

    count_ = condition_->apply() ? count_ + 1 : 0;

    if (log)
    {
        Info<< "    " << type() << ": " << name_ << " held for "
            << count_ << " of " << nIter_ << " consecutive iterations" << nl;
    }

    return count_ >= nIter_;
}


void Foam::functionObjects::runTimeControls::consecutiveCondition::write()
{
    condition_->write();

    this->conditionDict().set("consecutiveCount", count_);
}


void Foam::functionObjects::runTimeControls::consecutiveCondition::reset()
{
    condition_->reset();

    count_ = 0;
}


// ************************************************************************* //

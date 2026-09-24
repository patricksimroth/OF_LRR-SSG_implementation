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

Class
    Foam::functionObjects::runTimeControls::consecutiveCondition

Description
    runTimeControl condition that wraps another condition and is satisfied
    only once the wrapped condition has held for nIter consecutive calls.
    A single call where the wrapped condition fails resets the count.

    Usage
    \verbatim
    myCondition
    {
        type        consecutive;
        nIter       100;
        groupID     1;
        condition
        {
            type    average;
            ...
        }
    }
    \endverbatim

SourceFiles
    consecutiveCondition.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_functionObjects_runTimeControls_consecutiveCondition_H
#define Foam_functionObjects_runTimeControls_consecutiveCondition_H

#include "runTimeCondition.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
namespace runTimeControls
{

/*---------------------------------------------------------------------------*\
                    Class consecutiveCondition Declaration
\*---------------------------------------------------------------------------*/

class consecutiveCondition
:
    public runTimeCondition
{
protected:

    // Protected Data

        //- Number of consecutive calls the wrapped condition must hold
        label nIter_;

        //- Current number of consecutive calls it has held
        label count_;

        //- The wrapped condition
        autoPtr<runTimeCondition> condition_;


public:

    //- Runtime type information
    TypeName("consecutive");

    //- Constructor
    consecutiveCondition
    (
        const word& name,
        const objectRegistry& obr,
        const dictionary& dict,
        stateFunctionObject& state
    );

    //- Destructor
    virtual ~consecutiveCondition() = default;


    // Public Member Functions

        //- Apply the condition
        virtual bool apply();

        //- Write
        virtual void write();

        //- Reset
        virtual void reset();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace runTimeControls
} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

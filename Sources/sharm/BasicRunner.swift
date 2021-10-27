//
//  BasicRunner.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

class BasicRunner {

    let code: [Op]

}

private struct BasicNondeterminism: Nondeterminism {

    func chooseIndex(_ values: Set) -> Int {
        Int.random(in: 0..<values.count)
    }

    func chooseContext(_ context: Bag<Context>) -> Bag<Context>.Index {
        context.randomIndex
    }

}

private struct BasicRunnerOpVisitor: DeterministicContextOpVisitor {

    var context: Context
    var nondeterminism: BasicNondeterminism
    var vars: Dict

    mutating func choose() throws {
        try OpImpl.choose(context: &context, nondeterminism: nondeterminism)
    }

    mutating func store(_ addressOrNil: Value?) throws {
        try OpImpl.store(context: &context, vars: &vars, addressOrNil)
    }

    mutating func storeAddress() throws {
        <#code#>
    }

    mutating func load(_ value: Value?) throws {
        <#code#>
    }

    mutating func atomicInc(lazy: Bool) throws {
        <#code#>
    }

    mutating func atomicDec() throws {
        <#code#>
    }

    mutating func readonlyInc() throws {
        <#code#>
    }

    mutating func readonlyDec() throws {
        <#code#>
    }

    mutating func assertOp() throws {
        <#code#>
    }

    mutating func spawn(eternal: Bool) throws {
        <#code#>
    }

    mutating func apply() throws {
        <#code#>
    }

}

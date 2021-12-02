//
//  Op.swift
//  
//
//  Created by William Ma on 10/26/21.
//

enum OpError: Error {

    case typeMismatch(expected: Set<ValueType>, actual: [ValueType])
    case assertionFailure
    case unimplemented(String?)
    case stackIsEmpty
    case contextIsReadonly
    case contextIsAtomic
    case contextIsNotAtomic
    case stackTypeMismatch(expected: ValueType)
    case invalidAddress(address: Value)
    case unknownVar(varName: String)
    case varTypeMismatch(varName: String, expected: ValueType)
    case invalidCalltype(Int)
    case setIsEmpty
    case invalidKey(key: Value)
    case wrongCountSplit(actual: Int, expected: Int)
    case dictIsEmpty

}

enum Op: Hashable {

    case frame(name: String, params: VarTree)
    case push(value: Value)
    case sequential
    case choose
    case store(address: Value?)
    case storeVar(varTree: VarTree?)
    case jump(pc: Int)
    case jumpCond(pc: Int, cond: Value)
    case loadVar(varName: String?)
    case load(address: Value?)
    case address
    case nary(nary: Nary)
    case atomicInc(lazy: Bool)
    case atomicDec
    case readonlyInc
    case readonlyDec
    case assertOp
    case delVar(varName: String?)
    case ret
    case spawn(eternal: Bool)
    case apply
    case pop
    case cut(setName: String, varTree: VarTree)
    case incVar(varName: String)
    case dup
    case split(count: Int)
    case move(offset: Int)
    case log

}

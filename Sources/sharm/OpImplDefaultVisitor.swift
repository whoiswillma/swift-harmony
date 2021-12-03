//
//  OpImplDefaultVisitor.swift
//  
//
//  Created by William Ma on 11/12/21.
//

import Foundation

protocol DefaultOpImplVisitor: OpVisitor where Output == Void {

    var context: Context { get set }

}

extension DefaultOpImplVisitor {

    mutating func frame(name: String, params: VarTree, _ input: Input) throws {
        try OpImpl.frame(context: &context, name: name, params: params)
    }

    mutating func jump(pc: Int, _ input: Input) throws {
        try OpImpl.jump(context: &context, pc: pc)
    }

    mutating func delVar(varName: String?, _ input: Input) throws {
        try OpImpl.delVar(context: &context, varName: varName)
    }

    mutating func loadVar(varName: String?, _ input: Input) throws {
        try OpImpl.loadVar(context: &context, varName: varName)
    }

    mutating func nary(nary: Nary, _ input: Input) throws {
        try OpImpl.nary(context: &context, nary: nary)
    }

    mutating func storeVar(varTree: VarTree?, _ input: Input) throws {
        try OpImpl.storeVar(context: &context, varTree: varTree)
    }

    mutating func ret(_ input: Input) throws {
        try OpImpl.ret(context: &context)
    }

    mutating func push(value: Value, _ input: Input) throws {
        try OpImpl.push(context: &context, value: value)
    }

    mutating func pop(_ input: Input) throws {
        try OpImpl.pop(context: &context)
    }

    mutating func jumpCond(pc: Int, cond: Value, _ input: Input) throws {
        try OpImpl.jumpCond(context: &context, pc: pc, cond: cond)
    }

    mutating func address(_ input: Input) throws {
        try OpImpl.address(context: &context)
    }

    mutating func sequential(_ input: Input) throws {
        try OpImpl.sequential(context: &context)
    }

    mutating func apply(_ input: Input) throws {
        try OpImpl.apply(context: &context)
    }

    mutating func readonlyInc(_ input: Input) throws {
        try OpImpl.readonlyInc(context: &context)
    }

    mutating func readonlyDec(_ input: Input) throws {
        try OpImpl.readonlyDec(context: &context)
    }

    mutating func atomicInc(lazy: Bool, _ input: Input) throws {
        try OpImpl.atomicInc(context: &context, lazy: lazy)
    }

    mutating func atomicDec(_ input: Input) throws {
        try OpImpl.atomicDec(context: &context)
    }

    mutating func assertOp(_ input: Input) throws {
        try OpImpl.assertOp(context: &context)
    }

    mutating func cut(setName: String, key: VarTree?, value: VarTree, _ input: Input) throws {
        try OpImpl.cut(context: &context, setName: setName, key: key, value: value)
    }

    mutating func incVar(varName: String, _ input: Input) throws {
        try OpImpl.incVar(context: &context, varName: varName)
    }

    mutating func dup(_ input: Input) throws {
        try OpImpl.dup(context: &context)
    }

    mutating func split(count: Int, _ input: Input) throws {
        try OpImpl.split(context: &context, count: count)
    }

    mutating func move(offset: Int, _ input: Input) throws {
        try OpImpl.move(context: &context, offset: offset)
    }

    mutating func log(_ input: Input) throws -> Output {
        try OpImpl.log(context: &context)
    }

}

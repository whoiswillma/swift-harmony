//
//  OpVisitor.swift
//  
//
//  Created by William Ma on 10/29/21.
//

import Foundation

protocol OpVisitor {

    mutating func frame(name: String, params: VarTree) throws
    mutating func push(value: Value) throws
    mutating func sequential() throws
    mutating func choose() throws
    mutating func store(address: Value?) throws
    mutating func storeVar(varTree: VarTree?) throws
    mutating func jump(pc: Int) throws
    mutating func jumpCond(pc: Int, cond: Value) throws
    mutating func loadVar(varName: String?) throws
    mutating func load(address: Value?) throws
    mutating func address() throws
    mutating func nary(nary: Nary) throws
    mutating func atomicInc(lazy: Bool) throws
    mutating func atomicDec() throws
    mutating func readonlyInc() throws
    mutating func readonlyDec() throws
    mutating func assertOp() throws
    mutating func delVar(varName: String?) throws
    mutating func ret() throws
    mutating func spawn(eternal: Bool) throws
    mutating func apply() throws
    mutating func pop() throws
    mutating func cut(setName: String, varTree: VarTree) throws
    mutating func incVar(varName: String) throws
    mutating func dup() throws

}

extension Op {

    func accept<T: OpVisitor>(_ visitor: inout T) throws {
        switch self {
        case .frame(name: let name, params: let params):
            try visitor.frame(name: name, params: params)
        case .push(let value):
            try visitor.push(value: value)
        case .sequential:
            try visitor.sequential()
        case .choose:
            try visitor.choose()
        case .store(let address):
            try visitor.store(address: address)
        case .storeVar(let varTree):
            try visitor.storeVar(varTree: varTree)
        case .jump(pc: let pc):
            try visitor.jump(pc: pc)
        case .jumpCond(pc: let pc, cond: let cond):
            try visitor.jumpCond(pc: pc, cond: cond)
        case .loadVar(let varName):
            try visitor.loadVar(varName: varName)
        case .load(let address):
            try visitor.load(address: address)
        case .address:
            try visitor.address()
        case .nary(let nary):
            try visitor.nary(nary: nary)
        case .atomicInc(lazy: let lazy):
            try visitor.atomicInc(lazy: lazy)
        case .atomicDec:
            try visitor.atomicDec()
        case .readonlyInc:
            try visitor.readonlyInc()
        case .readonlyDec:
            try visitor.readonlyDec()
        case .assertOp:
            try visitor.assertOp()
        case .delVar(let varName):
            try visitor.delVar(varName: varName)
        case .ret:
            try visitor.ret()
        case .cut(let set, let value):
            try visitor.cut(setName: set, varTree: value)
        case .spawn(eternal: let eternal):
            try visitor.spawn(eternal: eternal)
        case .apply:
            try visitor.apply()
        case .pop:
            try visitor.pop()
        case .incVar(let varName):
            try visitor.incVar(varName: varName)
        case .dup:
            try visitor.dup()
        }
    }

}

protocol DeterministicContextOpVisitor: OpVisitor {

    var context: Context { get set }

}

extension DeterministicContextOpVisitor {

    mutating func frame(name: String, params: VarTree) throws {
        try OpImpl.frame(context: &context, name: name, params: params)
    }

    mutating func jump(pc: Int) throws {
        try OpImpl.jump(context: &context, pc: pc)
    }

    mutating func delVar(varName: String?) throws {
        try OpImpl.delVar(context: &context, varName: varName)
    }

    mutating func loadVar(varName: String?) throws {
        try OpImpl.loadVar(context: &context, varName: varName)
    }

    mutating func nary(nary: Nary) throws {
        try OpImpl.nary(context: &context, nary: nary)
    }

    mutating func storeVar(varTree: VarTree?) throws {
        try OpImpl.storeVar(context: &context, varTree: varTree)
    }

    mutating func ret() throws {
        try OpImpl.ret(context: &context)
    }

    mutating func push(value: Value) throws {
        try OpImpl.push(context: &context, value: value)
    }

    mutating func pop() throws {
        try OpImpl.pop(context: &context)
    }

    mutating func jumpCond(pc: Int, cond: Value) throws {
        try OpImpl.jumpCond(context: &context, pc: pc, cond: cond)
    }

    mutating func address() throws {
        try OpImpl.address(context: &context)
    }

    mutating func sequential() throws {
        try OpImpl.sequential(context: &context)
    }

    mutating func apply() throws {
        try OpImpl.apply(context: &context)
    }

    mutating func readonlyInc() throws {
        try OpImpl.readonlyInc(context: &context)
    }

    mutating func readonlyDec() throws {
        try OpImpl.readonlyDec(context: &context)
    }

    mutating func atomicInc(lazy: Bool) throws {
        try OpImpl.atomicInc(context: &context, lazy: lazy)
    }

    mutating func atomicDec() throws {
        try OpImpl.atomicDec(context: &context)
    }

    mutating func assertOp() throws {
        try OpImpl.assertOp(context: &context)
    }

    mutating func cut(setName: String, varTree: VarTree) throws {
        try OpImpl.cut(context: &context, setName: setName, varTree: varTree)
    }

    mutating func incVar(varName: String) throws {
        try OpImpl.incVar(context: &context, varName: varName)
    }

    mutating func dup() throws {
        try OpImpl.dup(context: &context)
    }

}

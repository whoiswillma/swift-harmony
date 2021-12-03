//
//  OpVisitor.swift
//  
//
//  Created by William Ma on 10/29/21.
//

protocol OpVisitor {

    associatedtype Input
    associatedtype Output

    mutating func frame(name: String, params: VarTree, _ input: Input) throws -> Output
    mutating func push(value: Value, _ input: Input) throws -> Output
    mutating func sequential(_ input: Input) throws -> Output
    mutating func choose(_ input: Input) throws -> Output
    mutating func store(address: Value?, _ input: Input) throws -> Output
    mutating func storeVar(varTree: VarTree?, _ input: Input) throws -> Output
    mutating func jump(pc: Int, _ input: Input) throws -> Output
    mutating func jumpCond(pc: Int, cond: Value, _ input: Input) throws -> Output
    mutating func loadVar(varName: String?, _ input: Input) throws -> Output
    mutating func load(address: Value?, _ input: Input) throws -> Output
    mutating func address(_ input: Input) throws -> Output
    mutating func nary(nary: Nary, _ input: Input) throws -> Output
    mutating func atomicInc(lazy: Bool, _ input: Input) throws -> Output
    mutating func atomicDec(_ input: Input) throws -> Output
    mutating func readonlyInc(_ input: Input) throws -> Output
    mutating func readonlyDec(_ input: Input) throws -> Output
    mutating func assertOp(_ input: Input) throws -> Output
    mutating func delVar(varName: String?, _ input: Input) throws -> Output
    mutating func ret(_ input: Input) throws -> Output
    mutating func spawn(eternal: Bool, _ input: Input) throws -> Output
    mutating func apply(_ input: Input) throws -> Output
    mutating func pop(_ input: Input) throws -> Output
    mutating func cut(setName: String, key: VarTree?, value: VarTree, _ input: Input) throws -> Output
    mutating func incVar(varName: String, _ input: Input) throws -> Output
    mutating func dup(_ input: Input) throws -> Output
    mutating func split(count: Int, _ input: Input) throws -> Output
    mutating func move(offset: Int, _ input: Input) throws -> Output
    mutating func log(_ input: Input) throws -> Output

}

protocol OpFunction: OpVisitor {

    func frame(name: String, params: VarTree, _ input: Input) -> Output
    func push(value: Value, _ input: Input) -> Output
    func sequential(_ input: Input) -> Output
    func choose(_ input: Input) -> Output
    func store(address: Value?, _ input: Input) -> Output
    func storeVar(varTree: VarTree?, _ input: Input) -> Output
    func jump(pc: Int, _ input: Input) -> Output
    func jumpCond(pc: Int, cond: Value, _ input: Input) -> Output
    func loadVar(varName: String?, _ input: Input) -> Output
    func load(address: Value?, _ input: Input) -> Output
    func address(_ input: Input) -> Output
    func nary(nary: Nary, _ input: Input) -> Output
    func atomicInc(lazy: Bool, _ input: Input) -> Output
    func atomicDec(_ input: Input) -> Output
    func readonlyInc(_ input: Input) -> Output
    func readonlyDec(_ input: Input) -> Output
    func assertOp(_ input: Input) -> Output
    func delVar(varName: String?, _ input: Input) -> Output
    func ret(_ input: Input) -> Output
    func spawn(eternal: Bool, _ input: Input) -> Output
    func apply(_ input: Input) -> Output
    func pop(_ input: Input) -> Output
    func cut(setName: String, key: VarTree?, value: VarTree, _ input: Input) -> Output
    func incVar(varName: String, _ input: Input) -> Output
    func dup(_ input: Input) -> Output
    func split(count: Int, _ input: Input) -> Output
    func move(offset: Int, _ input: Input) -> Output
    func log(_ input: Input) -> Output

}

extension Op {

    func accept<T: OpVisitor>(_ visitor: inout T, _ input: T.Input) throws -> T.Output {
        switch self {
        case .frame(name: let name, params: let params):
            return try visitor.frame(name: name, params: params, input)
        case .push(let value):
            return try visitor.push(value: value, input)
        case .sequential:
            return try visitor.sequential(input)
        case .choose:
            return try visitor.choose(input)
        case .store(let address):
            return try visitor.store(address: address, input)
        case .storeVar(let varTree):
            return try visitor.storeVar(varTree: varTree, input)
        case .jump(pc: let pc):
            return try visitor.jump(pc: pc, input)
        case .jumpCond(pc: let pc, cond: let cond):
            return try visitor.jumpCond(pc: pc, cond: cond, input)
        case .loadVar(let varName):
            return try visitor.loadVar(varName: varName, input)
        case .load(let address):
            return try visitor.load(address: address, input)
        case .address:
            return try visitor.address(input)
        case .nary(let nary):
            return try visitor.nary(nary: nary, input)
        case .atomicInc(lazy: let lazy):
            return try visitor.atomicInc(lazy: lazy, input)
        case .atomicDec:
            return try visitor.atomicDec(input)
        case .readonlyInc:
            return try visitor.readonlyInc(input)
        case .readonlyDec:
            return try visitor.readonlyDec(input)
        case .assertOp:
            return try visitor.assertOp(input)
        case .delVar(let varName):
            return try visitor.delVar(varName: varName, input)
        case .ret:
            return try visitor.ret(input)
        case .cut(let set, let key, let value):
            return try visitor.cut(setName: set, key: key, value: value, input)
        case .spawn(eternal: let eternal):
            return try visitor.spawn(eternal: eternal, input)
        case .apply:
            return try visitor.apply(input)
        case .pop:
            return try visitor.pop(input)
        case .incVar(let varName):
            return try visitor.incVar(varName: varName, input)
        case .dup:
            return try visitor.dup(input)
        case .split(count: let count):
            return try visitor.split(count: count, input)
        case .move(offset: let offset):
            return try visitor.move(offset: offset, input)
        case .log:
            return try visitor.log(input)
        }
    }

    func accept<T: OpFunction>(_ visitor: T, _ input: T.Input) -> T.Output {
        switch self {
        case .frame(name: let name, params: let params):
            return visitor.frame(name: name, params: params, input)
        case .push(let value):
            return visitor.push(value: value, input)
        case .sequential:
            return visitor.sequential(input)
        case .choose:
            return visitor.choose(input)
        case .store(let address):
            return visitor.store(address: address, input)
        case .storeVar(let varTree):
            return visitor.storeVar(varTree: varTree, input)
        case .jump(pc: let pc):
            return visitor.jump(pc: pc, input)
        case .jumpCond(pc: let pc, cond: let cond):
            return visitor.jumpCond(pc: pc, cond: cond, input)
        case .loadVar(let varName):
            return visitor.loadVar(varName: varName, input)
        case .load(let address):
            return visitor.load(address: address, input)
        case .address:
            return visitor.address(input)
        case .nary(let nary):
            return visitor.nary(nary: nary, input)
        case .atomicInc(lazy: let lazy):
            return visitor.atomicInc(lazy: lazy, input)
        case .atomicDec:
            return visitor.atomicDec(input)
        case .readonlyInc:
            return visitor.readonlyInc(input)
        case .readonlyDec:
            return visitor.readonlyDec(input)
        case .assertOp:
            return visitor.assertOp(input)
        case .delVar(let varName):
            return visitor.delVar(varName: varName, input)
        case .ret:
            return visitor.ret(input)
        case .cut(let set, let key, let value):
            return visitor.cut(setName: set, key: key, value: value, input)
        case .spawn(eternal: let eternal):
            return visitor.spawn(eternal: eternal, input)
        case .apply:
            return visitor.apply(input)
        case .pop:
            return visitor.pop(input)
        case .incVar(let varName):
            return visitor.incVar(varName: varName, input)
        case .dup:
            return visitor.dup(input)
        case .split(count: let count):
            return visitor.split(count: count, input)
        case .move(offset: let offset):
            return visitor.move(offset: offset, input)
        case .log:
            return visitor.log(input)
        }
    }

}

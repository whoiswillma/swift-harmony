//
//  OpVisitor.swift
//  
//
//  Created by William Ma on 10/29/21.
//

protocol OpVisitor {

    associatedtype Output

    mutating func frame(name: String, params: VarTree) throws -> Output
    mutating func push(value: Value) throws -> Output
    mutating func sequential() throws -> Output
    mutating func choose() throws -> Output
    mutating func store(address: Value?) throws -> Output
    mutating func storeVar(varTree: VarTree?) throws -> Output
    mutating func jump(pc: Int) throws -> Output
    mutating func jumpCond(pc: Int, cond: Value) throws -> Output
    mutating func loadVar(varName: String?) throws -> Output
    mutating func load(address: Value?) throws -> Output
    mutating func address() throws -> Output
    mutating func nary(nary: Nary) throws -> Output
    mutating func atomicInc(lazy: Bool) throws -> Output
    mutating func atomicDec() throws -> Output
    mutating func readonlyInc() throws -> Output
    mutating func readonlyDec() throws -> Output
    mutating func assertOp() throws -> Output
    mutating func delVar(varName: String?) throws -> Output
    mutating func ret() throws -> Output
    mutating func spawn(eternal: Bool) throws -> Output
    mutating func apply() throws -> Output
    mutating func pop() throws -> Output
    mutating func cut(setName: String, varTree: VarTree) throws -> Output
    mutating func incVar(varName: String) throws -> Output
    mutating func dup() throws -> Output
    mutating func split(count: Int) throws -> Output
    mutating func move(offset: Int) throws -> Output

}

protocol PureOpVisitor: OpVisitor {

    func frame(name: String, params: VarTree) -> Output
    func push(value: Value) -> Output
    func sequential() -> Output
    func choose() -> Output
    func store(address: Value?) -> Output
    func storeVar(varTree: VarTree?) -> Output
    func jump(pc: Int) -> Output
    func jumpCond(pc: Int, cond: Value) -> Output
    func loadVar(varName: String?) -> Output
    func load(address: Value?) -> Output
    func address() -> Output
    func nary(nary: Nary) -> Output
    func atomicInc(lazy: Bool) -> Output
    func atomicDec() -> Output
    func readonlyInc() -> Output
    func readonlyDec() -> Output
    func assertOp() -> Output
    func delVar(varName: String?) -> Output
    func ret() -> Output
    func spawn(eternal: Bool) -> Output
    func apply() -> Output
    func pop() -> Output
    func cut(setName: String, varTree: VarTree) -> Output
    func incVar(varName: String) -> Output
    func dup() -> Output
    func split(count: Int) -> Output
    func move(offset: Int) -> Output

}

extension Op {

    func accept<T: OpVisitor>(_ visitor: inout T) throws -> T.Output {
        switch self {
        case .frame(name: let name, params: let params):
            return try visitor.frame(name: name, params: params)
        case .push(let value):
            return try visitor.push(value: value)
        case .sequential:
            return try visitor.sequential()
        case .choose:
            return try visitor.choose()
        case .store(let address):
            return try visitor.store(address: address)
        case .storeVar(let varTree):
            return try visitor.storeVar(varTree: varTree)
        case .jump(pc: let pc):
            return try visitor.jump(pc: pc)
        case .jumpCond(pc: let pc, cond: let cond):
            return try visitor.jumpCond(pc: pc, cond: cond)
        case .loadVar(let varName):
            return try visitor.loadVar(varName: varName)
        case .load(let address):
            return try visitor.load(address: address)
        case .address:
            return try visitor.address()
        case .nary(let nary):
            return try visitor.nary(nary: nary)
        case .atomicInc(lazy: let lazy):
            return try visitor.atomicInc(lazy: lazy)
        case .atomicDec:
            return try visitor.atomicDec()
        case .readonlyInc:
            return try visitor.readonlyInc()
        case .readonlyDec:
            return try visitor.readonlyDec()
        case .assertOp:
            return try visitor.assertOp()
        case .delVar(let varName):
            return try visitor.delVar(varName: varName)
        case .ret:
            return try visitor.ret()
        case .cut(let set, let value):
            return try visitor.cut(setName: set, varTree: value)
        case .spawn(eternal: let eternal):
            return try visitor.spawn(eternal: eternal)
        case .apply:
            return try visitor.apply()
        case .pop:
            return try visitor.pop()
        case .incVar(let varName):
            return try visitor.incVar(varName: varName)
        case .dup:
            return try visitor.dup()
        case .split(count: let count):
            return try visitor.split(count: count)
        case .move(offset: let offset):
            return try visitor.move(offset: offset)
        }
    }

    func accept<T: PureOpVisitor>(_ visitor: T) -> T.Output {
        switch self {
        case .frame(name: let name, params: let params):
            return visitor.frame(name: name, params: params)
        case .push(let value):
            return visitor.push(value: value)
        case .sequential:
            return visitor.sequential()
        case .choose:
            return visitor.choose()
        case .store(let address):
            return visitor.store(address: address)
        case .storeVar(let varTree):
            return visitor.storeVar(varTree: varTree)
        case .jump(pc: let pc):
            return visitor.jump(pc: pc)
        case .jumpCond(pc: let pc, cond: let cond):
            return visitor.jumpCond(pc: pc, cond: cond)
        case .loadVar(let varName):
            return visitor.loadVar(varName: varName)
        case .load(let address):
            return visitor.load(address: address)
        case .address:
            return visitor.address()
        case .nary(let nary):
            return visitor.nary(nary: nary)
        case .atomicInc(lazy: let lazy):
            return visitor.atomicInc(lazy: lazy)
        case .atomicDec:
            return visitor.atomicDec()
        case .readonlyInc:
            return visitor.readonlyInc()
        case .readonlyDec:
            return visitor.readonlyDec()
        case .assertOp:
            return visitor.assertOp()
        case .delVar(let varName):
            return visitor.delVar(varName: varName)
        case .ret:
            return visitor.ret()
        case .cut(let set, let value):
            return visitor.cut(setName: set, varTree: value)
        case .spawn(eternal: let eternal):
            return visitor.spawn(eternal: eternal)
        case .apply:
            return visitor.apply()
        case .pop:
            return visitor.pop()
        case .incVar(let varName):
            return visitor.incVar(varName: varName)
        case .dup:
            return visitor.dup()
        case .split(count: let count):
            return visitor.split(count: count)
        case .move(offset: let offset):
            return visitor.move(offset: offset)
        }
    }

}

protocol DefaultOpVisitor: OpVisitor where Output == Void {

    var context: Context { get set }

}

extension DefaultOpVisitor {

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

    mutating func split(count: Int) throws {
        try OpImpl.split(context: &context, count: count)
    }

    mutating func move(offset: Int) throws {
        try OpImpl.move(context: &context, offset: offset)
    }

}

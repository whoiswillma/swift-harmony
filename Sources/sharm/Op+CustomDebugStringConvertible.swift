//
//  Op+CustomDebugStringConvertible.swift
//  
//
//  Created by William Ma on 11/2/21.
//

extension Op: CustomDebugStringConvertible {

    var debugDescription: String {
        switch self {
        case .frame(name: let name, params: let params):
            return "Op.frame(name: \(name.debugDescription), params: \(params.debugDescription))"
        case .push(value: let value):
            return "Op.push(value: \(value.debugDescription))"
        case .sequential:
            return "Op.sequential"
        case .choose:
            return "Op.choose"
        case .store(address: let address):
            return "Op.store(address: \(address.debugDescription))"
        case .storeVar(varTree: let varTree):
            return "Op.storeVar(varTree: \(varTree.debugDescription))"
        case .jump(pc: let pc):
            return "Op.jump(pc: \(pc))"
        case .jumpCond(pc: let pc, cond: let cond):
            return "Op.jumpCond(pc: \(pc), cond: \(cond.debugDescription))"
        case .loadVar(varName: let varName):
            return "Op.loadVar(varName: \(varName.debugDescription))"
        case .load(address: let address):
            return "Op.load(address: \(address.debugDescription))"
        case .address:
            return "Op.address"
        case .nary(nary: let nary):
            return "Op.nary(nary: \(String(reflecting: nary)))"
        case .atomicInc(lazy: let lazy):
            return "Op.atomicInc(lazy: \(lazy))"
        case .atomicDec:
            return "Op.atomicDec"
        case .readonlyInc:
            return "Op.readonlyInc"
        case .readonlyDec:
            return "Op.readonlyDec"
        case .assertOp:
            return "Op.assertOp"
        case .delVar(varName: let varName):
            return "Op.delVar(varName: \(varName.debugDescription))"
        case .ret:
            return "Op.ret"
        case .spawn(eternal: let eternal):
            return "Op.spawn(eternal: \(eternal))"
        case .apply:
            return "Op.apply"
        case .pop:
            return "Op.pop"
        case .cut(setName: let setName, key: let key, value: let value):
            return "Op.cut(setName: \(setName.debugDescription), key: \(key.debugDescription), value: \(value.debugDescription))"
        case .incVar(varName: let varName):
            return "Op.incVar(varName: \(String(reflecting: varName)))"
        case .dup:
            return "Op.dup"
        case .split(count: let count):
            return "Op.split(count: \(String(reflecting: count)))"
        case .move(offset: let offset):
            return "Op.move(offset: \(String(reflecting: offset)))"
        case .log:
            return "Op.log"
        }
    }

}

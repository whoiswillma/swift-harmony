//
//  Op+Decodable.swift
//  
//
//  Created by William Ma on 10/29/21.
//

import Foundation

extension Op {

    enum CodingKeys: String, CodingKey {
        case op
        case name
        case args
        case value
        case pc
        case arity
        case lazy
        case eternal
        case cond
        case set
    }

}

extension Op: Decodable {

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let op = try values.decode(String.self, forKey: .op)

        switch op {
        case "Frame":
            let name = try values.decode(String.self, forKey: .name)
            let params = try values.decode(String.self, forKey: .args)
            self = .frame(name: name, params: VarTree(string: params)!)

        case "Jump":
            let pc = Int(try values.decode(String.self, forKey: .pc))!
            self = .jump(pc: pc)

        case "DelVar":
            if values.contains(.value) {
                let value = try values.decode(String.self, forKey: .value)
                self = .delVar(varName: value)
            } else {
                self = .delVar(varName: nil)
            }

        case "LoadVar":
            if values.contains(.value) {
                let value = try values.decode(String.self, forKey: .value)
                self = .loadVar(varName: value)
            } else {
                self = .loadVar(varName: nil)
            }

        case "Nary":
            let value = try values.decode(String.self, forKey: .value)
            let arity = try values.decode(Int.self, forKey: .arity)

            var nary: Nary?
            switch value {
            case "+":
                nary = .plus(arity: arity)

            case "-":
                switch arity {
                case 2:
                    nary = .minus
                case 1:
                    nary = .negate
                default:
                    break
                }

            case "not":
                nary = .not

            case "==":
                nary = .equals

            case "atLabel":
                nary = .atLabel

            case "DictAdd":
                nary = .dictAdd

            case "..":
                nary = .range

            case "IsEmpty":
                nary = .isEmpty

            case "len":
                nary = .len

            case "SetAdd":
                nary = .setAdd

            case "keys":
                nary = .keys

            case "<":
                nary = .lessThan

            case ">=":
                nary = .greaterThanOrEqual

            case "in":
                nary = .in

            case "min":
                nary = .min

            case "max":
                nary = .max

            case ">":
                nary = .greaterThan

            case "<=":
                nary = .lessThanOrEqual

            case "|":
                nary = .union(arity: arity)

            case "get_context":
                nary = .getContext

            case "!=":
                nary = .notEquals

            case "*":
                nary = .times(arity: arity)

            case "%":
                nary = .mod

            default:
                break
            }

            guard let nary = nary else { fatalError() }
            assert(nary.arity == arity)
            self = .nary(nary: nary)

        case "StoreVar":
            if values.contains(.value) {
                let value = try values.decode(String.self, forKey: .value)
                self = .storeVar(varTree: VarTree(string: value)!)
            } else {
                self = .storeVar(varTree: nil)
            }

        case "Return":
            self = .ret

        case "Push":
            let value = try values.decode(Value.self, forKey: .value)
            self = .push(value: value)

        case "Apply":
            self = .apply

        case "Pop":
            self = .pop

        case "Load":
            if values.contains(.value) {
                let values = try values.decode([Value].self, forKey: .value)
                self = .load(address: .address(values))
            } else {
                self = .load(address: nil)
            }

        case "Sequential":
            self = .sequential

        case "Store":
            if values.contains(.value) {
                let indexPath = try values.decode([Value].self, forKey: .value)
                self = .store(address: .address(indexPath))
            } else {
                self = .store(address: nil)
            }

        case "Choose":
            self = .choose

        case "JumpCond":
            let pc = Int(try values.decode(String.self, forKey: .pc))!
            let cond = try values.decode(Value.self, forKey: .cond)
            self = .jumpCond(pc: pc, cond: cond)

        case "Address":
            self = .address

        case "AtomicInc":
            let lazy = try values.decode(String.self, forKey: .lazy)
            assert(lazy == "True" || lazy == "False")
            self = .atomicInc(lazy: lazy == "True")

        case "AtomicDec":
            self = .atomicDec

        case "ReadonlyInc":
            self = .readonlyInc

        case "ReadonlyDec":
            self = .readonlyDec

        case "Assert":
            self = .assertOp

        case "Spawn":
            let eternal = try values.decode(String.self, forKey: .eternal)
            assert(eternal == "True" || eternal == "False")
            self = .spawn(eternal: eternal == "True")

        case "Cut":
            let set = try values.decode(String.self, forKey: .set)
            let value = try values.decode(String.self, forKey: .value)
            self = .cut(setName: set, varTree: VarTree(string: value)!)

        case "IncVar":
            let varName = try values.decode(String.self, forKey: .value)
            self = .incVar(varName: varName)

        case "Dup":
            self = .dup

        default:
            fatalError(op)
        }
    }

}

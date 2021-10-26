//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

enum Nary: String, Hashable {

    case minus
    case not
    case equals
    case dictAdd
    case plus

    var arity: Int {
        switch self {
        case .minus: return 2
        case .not: return 1
        case .equals: return 2
        case .dictAdd: return 3
        case .plus: return 2
        }
    }

}

enum Op: Hashable {

    case frame(name: String, params: [String])
    case push(Value)
    case sequential
    case choose
    case store(Value)
    case storeAddress
    case storeVar(String)
    case jump(pc: Int)
    case jumpCond(pc: Int, cond: Value)
    case loadVar(String)
    case address
    case nary(Nary)
    case atomicInc(lazy: Bool)
    case atomicDec
    case readonlyInc
    case readonlyDec
    case assertOp
    case delVar(String)
    case ret
    case spawn(eternal: Bool)
    case apply
    case pop

}

extension Op: Decodable {

    enum CodingKeys: String, CodingKey {
        case op
        case name
        case args
        case value
        case pc
        case arity
        case lazy
        case eternal
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let op = try values.decode(String.self, forKey: .op)

        switch op {
        case "Frame":
            let name = try values.decode(String.self, forKey: .name)
            let params = try values.decode(String.self, forKey: .args)
            if params == "()" {
                self = .frame(name: name, params: [])
            } else if params.starts(with: "(") {
                let afterFirst = params.index(after: params.startIndex)
                let beforeLast = params.index(before: params.endIndex)
                assert(params[beforeLast] == ")")
                let commaSeparated = params[afterFirst..<beforeLast]
                    .split(separator: ",")
                    .map {
                        $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                self = .frame(name: name, params: commaSeparated)
            } else {
                // just one param
                self = .frame(name: name, params: [params])
            }

        case "Jump":
            let pc = Int(try values.decode(String.self, forKey: .pc))!
            self = .jump(pc: pc)

        case "DelVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .delVar(value)

        case "LoadVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .loadVar(value)

        case "Nary":
            let value = try values.decode(String.self, forKey: .value)

            let nary: Nary
            switch value {
            case "+":
                nary = .plus

            default:
                fatalError(value)
            }

            let arity = try values.decode(Int.self, forKey: .arity)
            assert(nary.arity == arity)
            self = .nary(nary)

        case "StoreVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .storeVar(value)

        case "Return":
            self = .ret

        case "Push":
            let value = try values.decode(Value.self, forKey: .value)
            self = .push(value)

        case "Apply":
            self = .apply

        case "Pop":
            self = .pop

        default:
            fatalError(op)
        }
    }

}

extension Op {

    func apply(_ context: inout Context) {
        switch self {
        case let .frame(name: _, params: params):
            guard case let .dict(args) = context.stack.last! else { fatalError() }

            // save the current vars
            context.stack.append(.dict(context.vars))
            context.stack.append(.int(context.fp))

            // match args with params
            context.vars = Dict()
            assert(args.count == params.count)
            for i in 0..<params.count {
                context.vars[.atom(params[i])] = args.elements[i].value
            }

            context.fp = context.stack.count
            context.pc += 1

        case let .jump(pc: pc):
            context.pc = pc

        case let .delVar(varName):
            context.vars[.atom(varName)] = nil
            context.pc += 1

        case let .loadVar(varName):
            let value = context.vars[.atom(varName)]!
            context.stack.append(value)
            context.pc += 1

        case let .nary(nary):
            switch nary {
            case .plus:
                let rhs = context.stack.popLast()
                let lhs = context.stack.popLast()

                switch (lhs, rhs) {
                case let (.int(lhs), .int(rhs)):
                    context.stack.append(.int(lhs + rhs))

                default:
                    fatalError()
                }

            default:
                fatalError()
            }

            context.pc += 1

        case let .storeVar(varName):
            let value = context.stack.popLast()!
            context.vars[.atom(varName)] = value
            context.pc += 1

        case .ret:
            let result = context.vars[.atom("result")] ?? .noneVal

            guard case let .int(originalFp) = context.stack.popLast()! else { fatalError() }
            context.fp = originalFp

            guard case let .dict(originalVars) = context.stack.popLast()! else { fatalError() }
            context.vars = originalVars

            // pop call arguments
            _ = context.stack.popLast()!

            if context.stack.isEmpty {
                context.terminated = true
                return
            }

            guard case let .pc(returnPc) = context.stack.popLast()! else { fatalError() }
            context.pc = returnPc

            context.stack.append(result)

        case let .push(value):
            context.stack.append(value)
            context.pc += 1

        case .apply:
            let args = context.stack.popLast()!
            let f = context.stack.popLast()!

            switch f {
            case let .dict(dict):
                context.stack.append(dict[args]!)

            case let .pc(pc):
                context.stack.append(.pc(context.pc + 1))
                context.stack.append(args)
                context.pc = pc

            default:
                fatalError()
            }

        case .pop:
            _ = context.stack.popLast()!
            context.pc += 1

        default:
            fatalError()
        }
    }

}

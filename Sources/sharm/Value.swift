//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Collections

typealias Dict = OrderedDictionary<Value, Value>

indirect enum Value: Hashable {

    case noneVal
    case atom(String)
    case bool(Bool)
    case int(Int)
    case dict(Dict)
    case address(Value)
    case pc(Int)
    case set(Set<Value>)

}

extension Value: Decodable {

    enum CodingKeys: String, CodingKey {
        case type
        case value
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let type = try values.decode(String.self, forKey: .type)

        switch type {
        case "atom":
            let value = try values.decode(String.self, forKey: .value)
            self = .atom(value)

        case "bool":
            let value = try values.decode(String.self, forKey: .value)
            assert(value == "True" || value == "False")
            self = .bool(value == "True")

        case "int":
            let value = Int(try values.decode(String.self, forKey: .value))!
            self = .int(value)

        case "dict":
            let dictEntries = try values.decode([[String: Value]].self, forKey: .value)
            var dict = Dict()
            for entry in dictEntries {
                let key = entry["key"]!
                let value = entry["value"]!
                dict[key] = value
            }
            self = .dict(dict)

        case "address":
            let value = try values.decode(Value.self, forKey: .value)
            self = .address(value)

        case "pc":
            let value = Int(try values.decode(String.self, forKey: .value))!
            self = .pc(value)

        case "set":
            let value = Set(try values.decode([Value].self, forKey: .value))
            self = .set(value)

        default:
            fatalError()
        }
    }

}

extension Value: CustomStringConvertible {

    var description: String {
        switch self {
        case .noneVal: return "None"
        case .atom(let value): return ".\(value)"
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return "\(value)"
        case .dict(let value): return "{\(value.map { "\($0):\($1)" }.joined(separator: ", "))}"
        case .address(let value): return "?\(value)"
        case .pc(let value): return "PC(\(value))"
        case .set(let value): return "{\(value.map { "\($0)" }.joined(separator: ", "))}"
        }
    }

}

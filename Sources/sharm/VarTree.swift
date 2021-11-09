//
//  VarTree.swift
//  
//
//  Created by William Ma on 10/29/21.
//

indirect enum VarTree: Hashable {

    case name(String)
    case tuple([VarTree])

}

extension VarTree: CustomStringConvertible {

    var description: String {
        switch self {
        case .name(let name): return name
        case .tuple(let elems): return "(\(elems.map(\.description).joined(separator: ",")))"
        }
    }

}

extension VarTree {

    init?(string: String) {
        var tokens = VarTree.tokenize(string)
        guard let tree = VarTree.parse(&tokens) else { return nil }
        if tokens.isEmpty {
            self = tree
        } else {
            return nil
        }
    }

    private enum Token: Equatable {
        case lparen
        case id(String)
        case comma
        case rparen
    }

    private static func tokenize(_ str: String) -> [Token] {
        let special: Set<Character> = ["(", ",", ")"]
        var str = str.components(separatedBy: .whitespacesAndNewlines).joined()
        var tokens: [Token] = []
        while !str.isEmpty {
            switch str.first {
            case "(":
                tokens.append(.lparen)
                str.removeFirst()

            case ",":
                tokens.append(.comma)
                str.removeFirst()

            case ")":
                tokens.append(.rparen)
                str.removeFirst()

            default:
                let end = str.firstIndex(where: special.contains) ?? str.endIndex
                tokens.append(.id(String(str[..<end])))
                str.removeSubrange(..<end)
            }
        }

        return tokens
    }

    private static func parse(_ tokens: inout [Token]) -> VarTree? {
        switch tokens.first {
        case .lparen: return parseTuple(&tokens)
        case .id: return parseId(&tokens)
        default: return nil
        }
    }

    private static func parseTuple(_ tokens: inout [Token]) -> VarTree? {
        guard case .lparen = tokens.first else { return nil }
        tokens.removeFirst()

        var elems = [VarTree]()
        while !tokens.isEmpty {
            if case .rparen = tokens.first {
                tokens.removeFirst()
                break
            }

            guard let node = parse(&tokens) else { return nil }
            elems.append(node)

            if tokens.first == .comma {
                tokens.removeFirst()
            } else {
                guard tokens.first == .rparen else { return nil }
            }
        }

        return .tuple(elems)
    }

    private static func parseId(_ tokens: inout [Token]) -> VarTree? {
        guard case .id(let name) = tokens.first else { return nil }

        tokens.removeFirst()
        return .name(name)
    }

}

extension VarTree: CustomDebugStringConvertible {

    var debugDescription: String {
        switch self {
        case .name(let name): return "VarTree.name(\(name.debugDescription))"
        case .tuple(let tuple): return "VarTree.tuple(\(tuple))"
        }
    }

}

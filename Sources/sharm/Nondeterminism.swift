//
//  Nondeterminism.swift
//  
//
//  Created by William Ma on 10/26/21.
//

protocol Nondeterminism {

    func chooseIndex(_ values: HSet) -> Int
    func chooseContext(_ context: [Context]) -> Int

}

class BookkeepingNondeterminism: Nondeterminism {

    enum History {
        case index(Int, HSet)
        case context(String, [String])
    }

    var history: [History] = []

    func chooseIndex(_ values: HSet) -> Int {
        let index = Int.random(in: 0..<values.count)
        history.append(.index(index, values))
        return index
    }

    func chooseContext(_ contexts: [Context]) -> Int {
        let index = Int.random(in: 0..<contexts.count)
        let name = contexts[index].name
        history.append(.context(name, contexts.map(\.name)))
        return index
    }

}

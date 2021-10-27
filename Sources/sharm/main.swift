#!/usr/bin/env swift

import Foundation
import OrderedCollections

struct SimpleNondeterminism: Nondeterminism {

    func chooseIndex(_ values: Set) -> Int {
        Int.random(in: 0..<values.count)
    }

    func chooseContext(_ context: Bag<Context>) -> Bag<Context>.Index {
        context.startIndex
    }

}

struct State {

    var nondeterminism: Nondeterminism
    var contexts: Bag<Context>
    var current: Bag<Context>.Index
    var vars: Dict

    init(nondeterminism: Nondeterminism) {
        self.nondeterminism = nondeterminism
        self.contexts = Bag()
        self.current = self.contexts.add(Context())
        self.vars = Dict()
    }

    func nonterminated() -> Swift.Set<Context> {
        contexts.elements().filter { !$0.terminated }
    }

    func allTerminated() -> Bool {
        nonterminated().isEmpty
    }

}

struct HVM: Decodable {
    let code: [Op]
}

let url = URL(fileURLWithPath: "/Users/williamma/Documents/sharm/Peterson.hvm")
let hvmData = try Data(contentsOf: url)
let hvm = try JSONDecoder().decode(HVM.self, from: hvmData)

var state = State(
    nondeterminism: SimpleNondeterminism()
)
while !state.allTerminated() {
    let index = state.nondeterminism.chooseContext(state.contexts)
    state.current = index
    let current = state.contexts.get(index: index)
    print(hvm.code[current.pc])
    assert(hvm.code[current.pc].apply(to: &state))
}

#!/usr/bin/env swift

import Foundation
import OrderedCollections

protocol Nondeterminism {

    mutating func chooseIndex(_ values: [Value]) -> Int
    mutating func chooseIndex(_ context: [Context]) -> Int

}

struct State {

    var nondeterminism: Nondeterminism
    var contexts: Bag<Context>
    var current: Bag<Context>.Index
    var vars: Dict

}

struct HVM: Decodable {
    let code: [Op]
}

let url = URL(fileURLWithPath: "/Users/williamma/Documents/sharm/Peterson.hvm")
let hvmData = try Data(contentsOf: url)
let hvm = try JSONDecoder().decode(HVM.self, from: hvmData)

var context = Context()
while !context.terminated {
    print(context)
    print(hvm.code[context.pc])
    hvm.code[context.pc].apply(&context)
}
print(context)

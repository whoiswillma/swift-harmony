//
//  State.swift
//  
//
//  Created by William Ma on 11/11/21.
//

import Foundation

struct State: Hashable {

    static let initialState = State(
        contextBag: Bag([.initContext]),
        vars: HDict(),
        nextContextToRun: .initContext
    )

    var contextBag: Bag<Context>
    var vars: HDict

    var nextContextToRun: Context

    var nonterminatedContexts: [Context] {
        contextBag.elements().filter { !$0.terminated }
    }

    var allTerminated: Bool {
        nonterminatedContexts.isEmpty
    }

    var runnable: [Context] {
        let contexts = nonterminatedContexts

        if let context = contexts.first(where: { $0.isAtomic }) {
            return [context]
        } else {
            return contexts
        }
    }

}

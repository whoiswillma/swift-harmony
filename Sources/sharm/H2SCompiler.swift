//
//  H2SCompiler.swift
//  
//
//  Created by William Ma on 11/2/21.
//

import Foundation

class H2SCompiler {

    private let code: [Op]
    private let sharmSourcesDir: String
    private let writer: H2SUtil.FileWriter
    private let genSanityChecks: Bool

    init(
        code: [Op],
        sharmSourcesDir: String,
        outputDir: String,
        dryRun: Bool,
        genSanityChecks: Bool
    ) {
        self.code = code
        self.sharmSourcesDir = sharmSourcesDir
        self.writer = H2SUtil.FileWriter(outputDir: outputDir, dryRun: dryRun)
        self.genSanityChecks = genSanityChecks
    }

    func run() throws {
        let orderedCollectionsSwift = try H2SUtil.generateOrderedCollectionSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(orderedCollectionsSwift, filename: "OrderedCollections.swift")

        let sharmSwift = try H2SUtil.generateSharmSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(sharmSwift, filename: "Sharm.swift")

        let mainSwift = generateMainSwift()
        try writer.write(mainSwift, filename: "main.swift")
    }

    private func generateMainSwift() -> String {
        generatePreamble()
        + H2SUtil.generateStepFunction(
            ops: code,
            genLine: H2SCompilerLineGenerator(),
            enableAssertions: genSanityChecks,
            printContext: genSanityChecks,
            printVars: genSanityChecks
        )
        + generateMain()
    }

    private func generatePreamble() -> String {
        return """
        var vars = HDict()
        var contextBag = Bag<Context>([.initContext])
        var threadCount: Int = 0

        func nonterminatedContexts() -> Set<Context> {
            contextBag.elements().filter { !$0.terminated }
        }

        func getRunnable() -> [Context] {
            let contexts = nonterminatedContexts()

            let atomicContexts = contexts.filter { $0.isAtomic }
            assert(atomicContexts.count <= 1)

            if let context = atomicContexts.first {
                return [context]
            } else {
                return contexts.sorted(by: { $0.name < $1.name })
            }
        }


        """
    }

    private func generateMain() -> String {
        return """
        while let context = getRunnable().randomElement() {
            var newContext = context
            try step(context: &newContext)
            contextBag.remove(context)
            contextBag.add(newContext)
        }

        """
    }

}

private struct H2SCompilerLineGenerator: H2SDefaultGenLineVisitor {

    func spawn(eternal: Bool) -> String {
        #"""
        try contextBag.add(OpImpl.spawn(parent: &context, name: "T\(threadCount)", eternal: \#(String(reflecting: eternal))))
        threadCount += 1
        """#
    }

}

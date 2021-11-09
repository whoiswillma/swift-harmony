//
//  H2SModelChecker.swift
//  
//
//  Created by William Ma on 11/9/21.
//

import Foundation

class H2SModelChecker {

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
        return ""
    }

}

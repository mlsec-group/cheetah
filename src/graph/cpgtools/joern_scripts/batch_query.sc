def query(method: Method, outFile: String): Unit = {
    val indexAccesses = method.ast.isIdentifier.where(_.astParent.isCallTo("<operator>.indirectIndexAccess")).l
    val buffers = indexAccesses.map(_.astSiblings.ast.isIdentifier.last).l
    val flows = buffers.reachableByFlows(method.parameter)

    flows.l.toJsonPretty #> outFile

    // alternatively you can output a table
    // flows.p #> outFile
}

/*
 *      MAIN
 */
def main(cpgFileArg: String, outFileArg: String): Unit = {
    val cpgFiles = cpgFileArg.split(",")
    val outFiles = outFileArg.split(",")

    println(s"Exporting ${cpgFiles.size} CPGs to ${outFiles.size} files")
    println(s"CPG files: ${cpgFiles.mkString(", ")}")
    println(s"Output files: ${outFiles.mkString(", ")}")
    (cpgFiles zip outFiles).foreach({
        case (cpgFile, outFile) =>
            importCpg(cpgFile)
            val methods = cpg.method
                .filterNot(_.isExternal)
                .filterNot(_.code == "<global>")
                .filterNot(_.astChildren.isBlock.code.headOption.getOrElse("<empty>") == "<empty>")
                .l
            if (methods.size > 0) {
                val m = methods.head
                query(m, outFile)
            }
            else {
                System.err.println(s"Target function not found in CPG (potential parser error) for ${cpgFile}")
            }
    })
}

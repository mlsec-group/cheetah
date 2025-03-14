import io.joern.dataflowengineoss.queryengine.{EngineContext, EngineConfig}

import scala.collection.parallel.CollectionConverters._


def stripAll(s: String, bad: String): String = {
    @scala.annotation.tailrec def start(n: Int): String = {
        if (n == s.length) ""
        else if (bad.indexOf(s.charAt(n)) < 0) end(n, s.length)
        else start(1 + n)
    }

    @scala.annotation.tailrec def end(a: Int, n: Int): String = {
        if (n <= a) s.substring(a, n)
        else if (bad.indexOf(s.charAt(n - 1)) < 0) s.substring(a, n)
        else end(a, n - 1)
    }
    start(0)
}


/*
 *      AST TRAVERSALS
*/
def toStatement(node: AstNode, belowAssignment: Boolean = false): AstNode = {
    val parent = node
        .iterator
        .collect {
        case n if (
                !n.astParent.isControlStructure &&
                !n.astParent.isBlock && 
                !n.astParent.isMethod &&
                !(belowAssignment && !n.astParent.isCallTo("<operator>.assignment").isEmpty)
            ) => n.iterator.repeat(_.astParent)(
                                _.until(_.filter(curr => (
                                    curr.astParent.isControlStructure ||
                                    curr.astParent.isBlock ||
                                    curr.astParent.isMethod ||
                                    (belowAssignment && !curr.astParent.isCallTo("<operator>.assignment").isEmpty)))))
                            .head
        case default => default
        }
        .l
    if(parent.size > 0) {
        return parent.head
    } else {
        return node
    }
}

def annotateNCS(method: Method)(implicit cpg: Cpg): Unit = {
    method.astChildren
          .isBlock
          .head
          .ast
          .whereNot(_.astChildren)
          .map(n => toStatement(n))
          .dedup
          .zipWithIndex
          .map((node, i) => node.iterator.newTagNodePair("ncs", i.toString).store())
          .l
}


def annotateComputedFrom(method: Method): Unit = {
    method.astChildren
          .isBlock
          .head
          .ast
          .whereNot(_.astChildren)
          .filter(n => toStatement(n).iterator.isCallTo("<operator>.assignment.*").size > 0)
          .filter(n => toStatement(n).iterator.isCall.argument(1).ast.map(_.id).contains(n.id))
          .map(n => n.iterator.newTagNodePair("dfg_c", toStatement(n).iterator.isCall.argument(2).head.id.toString).store())
          .l
}


def annotateLastWrite(method: Method): Unit = {
    method.astChildren
          .isBlock
          .head
          .ast
          .isIdentifier
          .map(n => (n, n.reachingDefIn
                         .whereNot(_.isBlock)
                         .whereNot(_.isMethod)
                         .filter(src => toStatement(src).isCallTo("<operator>.assignment.*").size > 0)
                         .filter(src => toStatement(src).iterator.isCall.argument(1).ast.map(_.id).contains(src.id))
                         .l))
          .filter((n, sources) => !sources.isEmpty)
          .map((n, sources) => n.iterator.newTagNodePair("dfg_w", sources.last.id.toString).store())
          .l
}


def annotateLastRead(method: Method): Unit = {
    method.astChildren
          .isBlock
          .head
          .ast
          .isIdentifier
          .map(n => (n, n.reachingDefIn
                         .whereNot(_.isBlock)
                         .whereNot(_.isMethod)
                         .filter(src => toStatement(src).isCallTo("<operator>.assignment.*").size > 0)
                         .filterNot(src => toStatement(src).iterator.isCall.argument(1).ast.map(_.id).contains(src.id))
                         .l))
          .filter((n, sources) => !sources.isEmpty)
          .map((n, sources) => n.iterator.newTagNodePair("dfg_r", sources.last.id.toString).store())
          .l
}

case class Graph(
    name: String,
    file: String,
    nodes: List[Node],
    ast: List[Edge],
    cfg: List[Edge],
    dfg: List[Edge]
)
case class Node(
    id: Long,
    ncs: Option[java.lang.Long],
    dfg_c: Option[java.lang.Long],
    dfg_r: Option[java.lang.Long],
    dfg_w: Option[java.lang.Long],
    code: String,
    label: String,
    typeFullName: Option[java.lang.String],
    lineNumber: Option[java.lang.Integer]
)
case class Edge(
    in: Long,
    out: Long
)

def getEdgesByLabel(root: AstNode, targetLabel: String): List[Edge] = {
    val astNodes = root.ast.id.l
    root.ast
        .map(n => n.outE.toList
                    .filter(_.label == targetLabel)
                    .map(_.src.id)
                    .map(other => (n.id, other))
                    .filter((in, out) => astNodes.contains(in) && astNodes.contains(out))
        )
        .l.flatten
        .map((in, out) => Edge(in, out))
}

def getType(n: AstNode): Option[String] = {
    val identifiers = n.iterator.isIdentifier.typeFullName.toList
    val literals = n.iterator.isLiteral.typeFullName.toList
    val locals = n.iterator.isLocal.typeFullName.toList
    (identifiers ++ literals ++ locals).headOption
}

def exportCpg(cpg: Cpg, outFile: String): Unit = {
    val m = cpg.method
               .filterNot(_.isExternal)
               .filterNot(_.code == "<global>")
               .head
    val b = m.astChildren
             .isBlock
             .head

    val nodes = b.ast.l.map(n => Node(
        n.id,
        n.tag.name("ncs").value.headOption.map(_.toLong),
        n.tag.name("dfg_c").value.headOption.map(_.toLong),
        n.tag.name("dfg_r").value.headOption.map(_.toLong),
        n.tag.name("dfg_w").value.headOption.map(_.toLong),
        n.code,
        n.label,
        getType(n),
        n.lineNumber.map(l => Integer.valueOf(l))
    ))
    val astEdges = getEdgesByLabel(b, "AST")
    val cfgEdges = getEdgesByLabel(b, "CFG")
    val dfgEdges = getEdgesByLabel(b, "REACHING_DEF")

    List(Graph(
        name=m.name,
        file=m.file.name.head,
        nodes=nodes,
        ast=astEdges,
        cfg=cfgEdges,
        dfg=dfgEdges
    )).toJsonPretty #> outFile
}


/*
 *      MAIN
 */
@main def exec(cpgFileArg: String, outFileArg: String) = {
    val cpgFiles = cpgFileArg.split(",")
    val outFiles = outFileArg.split(",")

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
                annotateNCS(m)
                annotateComputedFrom(m)
                annotateLastRead(m)
                annotateLastWrite(m)
                run.commit

                exportCpg(cpg, outFile)
            }
            else {
                System.err.println(s"Target function not found in CPG (potential parser error) for ${cpgFile}")
            }
    })
}

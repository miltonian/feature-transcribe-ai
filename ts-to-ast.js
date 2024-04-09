const ts = require("typescript");

function parseFile(filePath) {
  const fileContents = ts.sys.readFile(filePath);
  if (!fileContents) throw new Error(`Failed to read file: ${filePath}`);

  const sourceFile = ts.createSourceFile(
    filePath,
    fileContents,
    ts.ScriptTarget.Latest,
    true
  );

  function serializeAST(node, indentLevel = 0) {
    let serializedAst = "";
    function appendOutput(output) {
      serializedAst += output + "\n";
    }

    (function serialize(node, indentLevel = 0) {
      appendOutput(
        `${" ".repeat(indentLevel * 2)}${ts.SyntaxKind[node.kind]}: ${node
          .getText()
          .trim()}`
      );
      node.forEachChild((child) => serialize(child, indentLevel + 1));
    })(node);

    return serializedAst;
  }

  function processNode(node) {
    if (node.kind === ts.SyntaxKind.CallExpression) {
      const call = node;
      // Ensure the callee is an identifier named 'describe' or 'it'
      if (
        call.expression &&
        call.expression.kind === ts.SyntaxKind.Identifier
      ) {
        const calleeName = call.expression.escapedText;
        if (calleeName === "describe" || calleeName === "it") {
          const snippet = call.getText(sourceFile);
          const astSegment = serializeAST(call);
          console.log(
            JSON.stringify({ ast: astSegment.trim(), code: snippet.trim() })
          );
        }
      }
    } else {
      // Define the node kinds you're interested in, e.g., FunctionDeclaration
      const significantNodeKinds = [
        ts.SyntaxKind.FunctionDeclaration,
        ts.SyntaxKind.ClassDeclaration,
        ts.SyntaxKind.InterfaceDeclaration,
        ts.SyntaxKind.EnumDeclaration,
        ts.SyntaxKind.TypeAliasDeclaration,
        ts.SyntaxKind.VariableStatement,
        ts.SyntaxKind.ModuleDeclaration,
        ts.SyntaxKind.ExportAssignment,
        ts.SyntaxKind.ExportDeclaration,
        ts.SyntaxKind.ImportDeclaration,
      ];
      if (significantNodeKinds.includes(node.kind)) {
        const snippet = node.getText();
        const astSegment = serializeAST(node);
        console.log(
          JSON.stringify({ ast: astSegment.trim(), code: snippet.trim() })
        );
      }
    }

    ts.forEachChild(node, processNode);
  }

  processNode(sourceFile);
}

// Example usage:
const path = process.argv[2];
parseFile(path);

const ts = require("typescript");

function parseFile(filePath, search_value) {
  let i = 0;
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
    // console.log(node && node.code)
    // console.log(Object.keys(node))
    // console.log(node.getText())
    // const found = text.startsWith(search_value) && text
    // if(found){
      //   console.log(found)
      // }
      // const argument = node && node.arguments && node.arguments.map(a=>a.name && a.name.escapedText && console.log(a.name.escapedText))
      // console.log()
      // if (node && node.name && node.name.escapedText == search_value) {
    const text = node.getText(sourceFile)
    const found = text.startsWith(search_value) && text

    if (found) {
      const snippet = node.getText(sourceFile);
      const astSegment = serializeAST(node);
      console.log(
        JSON.stringify({
          ast: astSegment.trim(),
          // astJSON: JSON.stringify(node),
          code: snippet.trim(),
          path: filePath
          // routePath: const routePath = firstArg.text;,
        })
      );
    }
    // i++
    // if (search_value && node.kind === ts.SyntaxKind.CallExpression) {
    //   const call = node;
    //   // Ensure the callee is an identifier named 'describe' or 'it'
    //   if (call.arguments && call.arguments.length > 0) {
    //     const firstArg = call.arguments[0];
    //     // Check if the first argument is a string literal
    //     if (firstArg.kind === ts.SyntaxKind.StringLiteral) {
    //         const routePath = firstArg.text; // Get the text of the string literal
    //         // Check if the route path matches '/session/documents/:id'
    //         if (routePath === search_value) {
    //             // Here, we've found the specific route
    //             const snippet = call.getText(sourceFile);
    //             const astSegment = serializeAST(call);
    //             console.log(
    //                 JSON.stringify({ ast: astSegment.trim(), code: snippet.trim(), routePath })
    //             );
    //         }
    //     }
    // }
    //    if (
    //     call.expression &&
    //     (call.expression.kind === ts.SyntaxKind.Identifier)
    //   ) {
    //     const calleeName = call.expression.escapedText;
    //     if (calleeName === "describe" || calleeName === "it" ) {
    //       const snippet = call.getText(sourceFile);
    //       const astSegment = serializeAST(call);
    //       // console.log(
    //       //   JSON.stringify({ ast: astSegment.trim(), code: snippet.trim() })
    //       // );
    //     }
    //   }
    // } else {
    //   // Define the node kinds you're interested in, e.g., FunctionDeclaration
    //   const significantNodeKinds = [
    //     ts.SyntaxKind.FunctionDeclaration,
    //     ts.SyntaxKind.ClassDeclaration,
    //     ts.SyntaxKind.InterfaceDeclaration,
    //     ts.SyntaxKind.EnumDeclaration,
    //     ts.SyntaxKind.TypeAliasDeclaration,
    //     ts.SyntaxKind.VariableStatement,
    //     ts.SyntaxKind.ModuleDeclaration,
    //     ts.SyntaxKind.ExportAssignment,
    //     ts.SyntaxKind.ExportDeclaration,
    //     ts.SyntaxKind.ImportDeclaration,
    //   ];
    //   if (significantNodeKinds.includes(node.kind)) {
    //     const snippet = node.getText();
    //     const astSegment = serializeAST(node);
    //     // console.log(
    //     //   JSON.stringify({ ast: astSegment.trim(), code: snippet.trim() })
    //     // );
    //   }
    // }

    ts.forEachChild(node, processNode);
  }

  processNode(sourceFile);
}

// Example usage:
const path = process.argv[2];
const search_value = process.argv[3];
parseFile(path, search_value);

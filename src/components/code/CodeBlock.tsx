import { Highlight, themes } from "prism-react-renderer";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  language?: string;
  className?: string;
}

const CodeBlock = ({ code, language = "bash", className }: CodeBlockProps) => {
  return (
    <Highlight theme={themes.nightOwl} code={code.trim()} language={language}>
      {({ className: preClassName, style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className={cn(
            preClassName,
            "rounded-lg p-4 overflow-x-auto text-sm",
            className
          )}
          style={{
            ...style,
            backgroundColor: "hsl(var(--secondary) / 0.5)",
            margin: 0,
          }}
        >
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })} className="table-row">
              <span className="table-cell pr-4 text-muted-foreground/50 select-none text-right text-xs">
                {i + 1}
              </span>
              <span className="table-cell">
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </span>
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
};

export default CodeBlock;

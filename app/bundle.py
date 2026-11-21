"""A few ES modules of static/js joined into one classic script, for the HTML backup: a file opened from the disk
can not load modules from other files. Each module runs in its own function; its exports become an object the
modules after it read their imports from. Only the forms used here are understood:
    import { a, b as c } from './name.js';      export function / async function / class / const / let name
"""
import re
from pathlib import Path

JS_DIR = Path(__file__).parent / 'static' / 'js'
IMPORT = re.compile(r"^import\s*\{([^}]*)\}\s*from\s*'\./([\w-]+)\.js';[ \t]*$", re.M)
EXPORT = re.compile(r'^export\s+((?:async\s+)?function\*?|class|const|let)\s+([A-Za-z_$][\w$]*)', re.M)


class BundleError(Exception):
    pass


def module_code(name):
    src = (JS_DIR / f'{name}.js').read_text(encoding='utf-8')

    def imported(m):
        names = [n.strip() for n in m.group(1).split(',') if n.strip()]
        fields = ', '.join(n.replace(' as ', ': ') for n in names)
        return f"const {{ {fields} }} = modules['{m.group(2)}'];"

    src = IMPORT.sub(imported, src)
    exports = [m.group(2) for m in EXPORT.finditer(src)]
    src = EXPORT.sub(lambda m: f'{m.group(1)} {m.group(2)}', src)
    if re.search(r'^\s*(import|export)\b', src, re.M):
        raise BundleError(f'{name}.js uses an import or export form the bundler does not know')
    return f"modules['{name}'] = (() => {{\n'use strict';\n{src}\nreturn {{ {', '.join(exports)} }};\n}})();\n"


def bundle(names):
    """names in order: every module after the ones it imports."""
    body = ''.join(module_code(n) for n in names)
    return f'(() => {{\nconst modules = {{}};\n{body}}})();\n'

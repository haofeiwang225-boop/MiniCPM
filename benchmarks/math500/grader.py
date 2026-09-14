"""Compare extracted MATH-500 answers using LaTeX and symbolic equivalence.

这个文件的作用：判断"模型抽出来的答案"和"标准答案"在数学上是不是一回事。

为什么需要单独一个文件？
  数学题的答案写法五花八门：\frac12 和 0.5、90 和 90^\circ、(1,2) 和 \left(1,2\right)
  看起来不同但数学上相等。直接用字符串比较会误判，所以要用 sympy 做符号化简比较。

整体流程（三步）：
  1. normalize_latex：先把 LaTeX 写法统一一下（去括号、统一分数写法等）；
  2. _parse：把统一后的文本翻译成 sympy 能理解的数学对象；
  3. _equivalent：用 sympy 化简，判断两个对象是不是相等。

为什么 grade_answer 要开子进程？
  sympy.simplify 有时候会卡住（化简超时），放在子进程里可以限时杀掉，
  不会把整个评测流程拖死。
"""

import re
from dataclasses import dataclass
import json
import subprocess
import sys
from pathlib import Path

import sympy
from latex2sympy2 import latex2sympy, set_variances


def _brace_arguments(text):
    """给 \\frac 和 \\sqrt 等命令补全省略的花括号。

    为什么需要补全？
      LaTeX 允许 \\frac12 这样省略花括号（每个参数只取一个 token），
      但 latex2sympy2 解析器要求参数用花括号包裹，否则解析失败。
      所以要把 \\frac12 改成 \\frac{1}{2}。
    """
    # 从 pos 位置读取一个 TeX 参数，自动补全花括号。
    # 定义在内部是因为需要访问外层 text 并与 command 互相递归。
    def argument(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            raise ValueError('Missing TeX argument')
        if text[pos] == '{':
            # 已有花括号：按深度配对找到闭合括号，递归处理内部内容
            start, depth = pos + 1, 1
            pos += 1
            while pos < len(text):
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                if depth == 0:
                    return '{' + _brace_arguments(text[start:pos]) + '}', pos + 1
                pos += 1
            raise ValueError('Unbalanced TeX argument')
        if text[pos] == '\\':
            # 参数本身是个命令（如 \\frac\\pi2），交给 command 处理
            value, end = command(pos)
            return '{' + value + '}', end
        if text[pos] in '}[]':
            raise ValueError('Invalid TeX argument')
        # 单 token 参数：补上花括号，如 2 → {2}
        return '{' + text[pos] + '}', pos + 1

    # 解析一个 TeX 命令（如 \\frac、\\sqrt），递归处理其参数。
    # 定义在内部是因为需要访问外层 text 并与 argument 互相递归。
    def command(pos):
        # 匹配 \\命令名 或 \\单字符（如 \\, 间距命令）
        match = re.match(r'\\(?:[A-Za-z]+|.)', text[pos:])
        if match is None:
            raise ValueError('Incomplete TeX command')
        name = match[0]
        end = pos + len(name)
        if name == r'\frac':
            # 分数有两个参数：分子和分母
            numerator, end = argument(end)
            denominator, end = argument(end)
            return name + numerator + denominator, end
        if name == r'\sqrt':
            # 根号可选方括号索引（如 \\sqrt[3]{x}），再接一个参数
            while end < len(text) and text[end].isspace():
                end += 1
            index = ''
            if end < len(text) and text[end] == '[':
                close = text.find(']', end)
                if close < 0:
                    raise ValueError('Unclosed root index')
                index, end = text[end:close + 1], close + 1
            value, end = argument(end)
            return name + index + value, end
        return name, end

    result, pos = [], 0
    try:
        while pos < len(text):
            if text[pos] == '\\':
                value, pos = command(pos)
                result.append(value)
            else:
                result.append(text[pos])
                pos += 1
    except ValueError:
        return text  # 解析失败时保留原始文本，供后续字符串兜底比较
    return ''.join(result)


@dataclass(frozen=True)
class Assignment:
    """赋值语句，如 "x = (1, 2)" 或 "v = \\begin{pmatrix}1\\\\2\\end{pmatrix}"。

    label: 赋值左值（变量名或变量名元组）
    value: 赋值右值（数值、元组或矩阵）

    用 frozen=True 是因为赋值是不可变值对象，需要可哈希、可比较。
    判分时先比左值（变量名）再比右值（值）。
    """
    label: object
    value: object


def _assignment_label(value):
    """判断一个 sympy 对象是否可以作为赋值左值。

    可以作为左值：单个符号（如 x），或由不同符号组成的元组（如 (x, y)）。
    元组要求元素互不相同且都是符号，避免把数值元组误认为变量名元组。
    """
    return (isinstance(value, sympy.Symbol)
            or isinstance(value, sympy.Tuple) and len(set(value)) == len(value)
            and all(isinstance(item, sympy.Symbol) for item in value))


def normalize_latex(text: str) -> str:
    """把 LaTeX 答案的写法统一一下，方便后续解析。

    为什么要统一？
      同一个数学答案有很多种 LaTeX 写法，比如：
      - \tfrac{1}{2}、\dfrac{1}{2} 和 \frac{1}{2} 是一回事；
      - \left(1,2\right) 和 (1,2) 是一回事；
      - \boxed{0.5} 和 0.5 是一回事；
      - 90^\circ 和 90（度数标记）有时要忽略。
      不统一的话，后面解析会失败或误判。

    具体做了这些事：
      1. 去掉最外层的 \[ \] \( \) $ 这些"显示用"的定界符；
      2. 把 \tfrac、\dfrac 都改成 \frac；去掉 \left、\right（只是括号大小标记）；
      3. 把 \quad、\, 等各种"间距"命令换成普通空格；
      4. 把 \boxed{...} 和 \text{...} 的外壳剥掉，只留里面的内容；
      5. 去掉句末的句号（. 或 。）；单位由题干或标答约定处理；
      6. 给 \frac 和 \sqrt 补全花括号：\frac12 改成 \frac{1}{2}，否则解析器不认；
      7. 把多个连续空格压成一个。
    """
    text = text.strip()
    text = re.sub(r'^\*\*(.*?)\*\*([.。]?)$', r'\1\2', text, flags=re.DOTALL)
    for marker in (r'\[', r'\]', r'\(', r'\)', '$'):
        text = re.sub(r'(?<!\\)' + re.escape(marker), '', text)
    text = re.sub(r'\\(?:tfrac|dfrac)(?![A-Za-z])', r'\\frac', text)
    text = re.sub(r'\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|big|Big)(?![A-Za-z])', '', text)
    text = re.sub(r'\\(?:displaystyle|textstyle)\b', '', text)
    text = text.replace('{,}', ',')
    text = re.sub(
        r'\\(?:quad|qquad|enspace|thinspace|medspace|thickspace|negthinspace)\b'
        r'|(?<!\\)\\[,;!: ]|~', ' ', text)

    # Unwrap balanced boxes and text, preserving nested mathematical braces.
    while True:
        match = re.search(r'\\(?:boxed|text|mbox|mathrm)\s*\{', text)
        if match is None:
            break
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            if depth == 0:
                text = text[:match.start()] + text[start:end] + text[end + 1:]
                break
        else:
            break  # Malformed LaTeX remains available for the string fallback.

    # Strip sentence-ending punctuation only; preserve decimal points inside
    # numbers, tuple separators, signs, and factorial/exclamation operators.
    text = re.sub(r'(?<!\.)[.。]\s*$', '', text).strip()

    # Read balanced TeX arguments, including nested radicals and fractions.
    text = _brace_arguments(text)
    # Preserve command boundaries during parsing (e.g. \pi x is not \pix).
    return re.sub(r'\s+', ' ', text).strip()


def _tuple_parts(text):
    """把 (a, b, c) 这样的元组按顶层逗号拆成 [a, b, c]。

    为什么要单独写这个函数？
      Python 的 split(',') 会把 "(1,2),(3,4)" 错切成四段。
      这里用括号深度计数，只在"最外层"的逗号处切分，
      嵌套括号里的逗号不动。比如 (1, (2,3)) 切成 ['1', '(2,3)']。

    返回 None 表示这不是一个合法的元组（比如没带外层括号）。
    """
    if not (text.startswith('(') and text.endswith(')')):
        return None
    parts = []
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    start = 1
    for index, char in enumerate(text[1:-1], 1):
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return None
        elif char == ',' and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if stack or not parts:
        return None
    parts.append(text[start:-1].strip())
    if not all(parts):
        raise ValueError('Empty tuple component')
    return parts


def _parse(text):
    """把一段 LaTeX 文本翻译成 sympy 能理解的数学对象。

    为什么要分这么多情况？
      latex2sympy 这个库只能处理普通的表达式，碰到矩阵、等式、集合、坐标元组
      就会出错或解析错。所以这里先识别特殊格式，自己处理；处理不了再交给库。

    处理顺序（从特殊到一般）：
      - 矩阵：\begin{pmatrix}...\end{pmatrix}，按 & 和 \\ 切行列，递归解析每个元素；
      - 等式：形如 "左边=右边"，构造 sympy.Eq 对象（注意要排除 <=、>=、!=）；
      - 集合：\{1,2,3\}，按逗号切分，构造 FiniteSet（无序集合）；
      - 度数：把 90^\circ 转成 90*pi/180 这种弧度表达式，sympy 才能算；
      - 元组：(1,2,3) 用 _tuple_parts 拆开后递归解析每个分量；
      - 兜底：以上都不是，就交给 latex2sympy 解析普通表达式。
    """
    text = text.strip()
    # Scalar multiplication of a matrix, preserving dimensions and every entry.
    matrix_start = text.find(r'\begin{')
    if matrix_start > 0 and '=' not in text:
        scalar = text[:matrix_start].strip()
        matrix_value = _parse(text[matrix_start:])
        scalar_value = _parse(scalar)
        if not isinstance(matrix_value, sympy.MatrixBase) or not isinstance(scalar_value, sympy.Expr):
            raise ValueError('Unsupported matrix product')
        return scalar_value * matrix_value
    matrix = re.fullmatch(
        r'\\begin\{(pmatrix|bmatrix|matrix)\}(.*?)\\end\{\1\}', text, re.DOTALL)
    if matrix:
        rows = re.split(r'\\\\(?:\s*\[[^\]]*\])?', matrix[2].strip())
        if rows and not rows[-1].strip():
            rows.pop()
        cells = [row.split('&') for row in rows]
        if not cells or len({len(row) for row in cells}) != 1:
            raise ValueError('Invalid matrix dimensions')
        if any(not cell.strip() for row in cells for cell in row):
            raise ValueError('Empty matrix cell')
        return sympy.ImmutableMatrix([[_parse(cell.strip()) for cell in row] for row in cells])
    # latex2sympy2 treats some equalities as assignments rather than Eq objects.
    if text.count('=') == 1 and not re.search(r'[<>!]=|\\(?:le|ge|ne)', text):
        left, right = text.split('=')
        # Vector font commands annotate the assigned variable, not its value.
        left = re.sub(r'\\(?:mathbf|vec)\s*(?:\{([A-Za-z])\}|([A-Za-z]))',
                      lambda m: m[1] or m[2], left)
        left_value, right_value = _parse(left), _parse(right)
        if _assignment_label(left_value) and isinstance(right_value, (sympy.Tuple, sympy.MatrixBase)):
            if isinstance(left_value, sympy.Tuple) and (
                    not isinstance(right_value, sympy.Tuple) or len(left_value) != len(right_value)):
                raise ValueError('Assignment shape mismatch')
            return Assignment(left_value, right_value)
        if isinstance(left_value, sympy.MatrixBase) and isinstance(right_value, sympy.MatrixBase):
            if not _equivalent(left_value, right_value):
                raise ValueError('Inconsistent matrix equality')
            return right_value
        return sympy.Eq(left_value, right_value, evaluate=False)
    if text.startswith(r'\{') and text.endswith(r'\}'):
        content = text[2:-2].strip()
        if not content:
            return sympy.FiniteSet()
        parts = _tuple_parts('(' + content + ')') or [content]
        values = []
        for part in parts:
            if part.count(r'\pm') == 1:
                values.extend((_parse(part.replace(r'\pm', '+')),
                               _parse(part.replace(r'\pm', '-'))))
            elif r'\pm' in part:
                raise ValueError('Ambiguous multiple plus-minus signs')
            else:
                values.append(_parse(part))
        return sympy.FiniteSet(*values)
    # Convert numeric degree arguments inside expressions to radians.
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:\^\s*(?:\\circ\b|\{\s*\\circ\s*\})|°)',
        lambda m: r'\left(' + m[1] + r'\cdot\frac{\pi}{180}\right)', text)
    # latex2sympy2 cannot directly parse coordinate tuples: parse each component.
    parts = _tuple_parts(text)
    if parts is not None:
        return sympy.Tuple(*(_parse(part) for part in parts))
    parts = _tuple_parts('(' + text + ')')
    if parts is not None:
        return sympy.Tuple(*(_parse(part) for part in parts))
    # The library retains assignment values globally; one expression must not
    # change the meaning of symbols in the other answer or a later question.
    set_variances({})
    return latex2sympy(text)


def _equivalent(prediction, gold):
    """判断两个 sympy 对象在数学上是不是相等。

    为什么不直接用 ==？
      sympy 的 == 是"结构相等"，不是"数学相等"。
      比如 (x-1)*(x+1) 和 x**2-1 用 == 比较是 False，但数学上相等。
      所以要用 simplify 化简后再比。

    不同类型分别处理：
      - 矩阵：形状得一样，然后逐个位置比；
      - 集合 {1,2} 和 {2,1}：集合无序，所以要把两边的元素配对匹配；
      - 等式 x=2 和 2：允许"单变量=常数"和常数直接比；两个等式则看两边差值是否只差常数倍；
      - 元组 (1,2) 和 (1,2)：长度一样且每个位置都相等（有序）；
      - 普通表达式：simplify(prediction - gold) == 0，化简后差为零即相等。
    """
    if isinstance(prediction, Assignment) or isinstance(gold, Assignment):
        if isinstance(prediction, Assignment) and isinstance(gold, Assignment):
            return prediction.label == gold.label and _equivalent(prediction.value, gold.value)
        assignment, value = (prediction, gold) if isinstance(prediction, Assignment) else (gold, prediction)
        return not assignment.value.free_symbols and _equivalent(assignment.value, value)
    # In particular, infinity minus infinity is NaN, not zero.
    if prediction == gold and prediction is not sympy.nan:
        return True
    if isinstance(prediction, sympy.Set) or isinstance(gold, sympy.Set):
        if not isinstance(prediction, sympy.Set) or not isinstance(gold, sympy.Set):
            return False
        if not isinstance(prediction, sympy.FiniteSet) or not isinstance(gold, sympy.FiniteSet):
            return sympy.simplify(sympy.SymmetricDifference(prediction, gold)).is_empty is True
    if isinstance(prediction, sympy.MatrixBase) or isinstance(gold, sympy.MatrixBase):
        return (isinstance(prediction, sympy.MatrixBase)
                and isinstance(gold, sympy.MatrixBase)
                and prediction.shape == gold.shape
                and all(_equivalent(a, b) for a, b in zip(prediction, gold)))
    if isinstance(prediction, sympy.FiniteSet) or isinstance(gold, sympy.FiniteSet):
        if not isinstance(prediction, sympy.FiniteSet) or not isinstance(gold, sympy.FiniteSet):
            return False
        # Finite sets are unordered; match mathematically equivalent members.
        remaining = list(gold)
        for item in prediction:
            for index, candidate in enumerate(remaining):
                if _equivalent(item, candidate):
                    remaining.pop(index)
                    break
            else:
                return False
        return not remaining
    if isinstance(prediction, sympy.Equality) or isinstance(gold, sympy.Equality):
        if not isinstance(prediction, sympy.Equality) or not isinstance(gold, sympy.Equality):
            equation, value = ((prediction, gold) if isinstance(prediction, sympy.Equality)
                               else (gold, prediction))
            # Only a single symbol explicitly assigned a constant can omit its label.
            for variable, constant in ((equation.lhs, equation.rhs), (equation.rhs, equation.lhs)):
                if (isinstance(variable, sympy.Symbol) and isinstance(constant, sympy.Expr)
                        and not constant.free_symbols and isinstance(value, sympy.Expr)
                        and not value.free_symbols):
                    return _equivalent(constant, value)
            if (not equation.free_symbols and isinstance(value, sympy.Expr)
                    and not value.free_symbols):
                return _equivalent(equation.lhs, equation.rhs) and _equivalent(equation.rhs, value)
            return False
        a = sympy.simplify(prediction.lhs - prediction.rhs)
        b = sympy.simplify(gold.lhs - gold.rhs)
        if a == 0 or b == 0:
            return a == b
        # Only constant nonzero scaling is accepted; do not cancel variable factors.
        ratio = sympy.simplify(a / b)
        return not ratio.free_symbols and ratio.is_finite is True and ratio.is_zero is False
    if isinstance(prediction, sympy.Tuple) or isinstance(gold, sympy.Tuple):
        return (isinstance(prediction, sympy.Tuple)
                and isinstance(gold, sympy.Tuple)
                and len(prediction) == len(gold)
                and all(_equivalent(a, b) for a, b in zip(prediction, gold)))
    return sympy.simplify(prediction - gold) == 0


# 匹配所有支持的单位单词：长度（inch/feet/cm/mm/km/m）、时间（sec/min/hr）、货币（dollar/cent）
_UNIT_WORD = r'(?:inches|inch|feet|foot|ft|centimeters?|cm|millimeters?|mm|kilometers?|km|meters?|m|seconds?|sec|s|minutes?|min|hours?|hr|dollars?|cents?)'
# 在 _UNIT_WORD 基础上扩展：支持 square/cubic 前缀（面积/体积）、幂次后缀（如 m^2）、复合单位（如 meters per second）
_UNIT = rf'(?:(?:square|cubic)\s+)?{_UNIT_WORD}(?:\s*\^\s*\{{?[23]\}}?)?(?:\s+per\s+{_UNIT_WORD})?'
# 单位别名到规范形式的映射，如 'inches' → 'inch', 'centimeters' → 'cm'，用于单位归一化比较
_UNIT_ALIASES = {
    'inches': 'inch', 'feet': 'foot', 'ft': 'foot',
    'centimeter': 'cm', 'centimeters': 'cm', 'millimeter': 'mm', 'millimeters': 'mm',
    'kilometer': 'km', 'kilometers': 'km', 'meter': 'm', 'meters': 'm',
    'second': 's', 'seconds': 's', 'sec': 's', 'minute': 'min', 'minutes': 'min',
    'hour': 'hr', 'hours': 'hr', 'dollars': 'dollar', 'cents': 'cent',
}


def _canonical_unit(unit):
    """将单位文本归一化为规范形式，便于比较。

    步骤：转小写 → 压缩空格 → "square X" 转为 "X^2"、"cubic X" 转为 "X^3"
    → 统一幂次写法 → 用 _UNIT_ALIASES 映射别名 → 去空格。
    同一单位有多种写法（如 "square meters" 和 "m^2"），不归一化就无法比较。
    """
    unit = re.sub(r'\s+', ' ', unit.lower()).strip()
    if unit.startswith('square '):
        unit = unit[7:] + '^2'
    elif unit.startswith('cubic '):
        unit = unit[6:] + '^3'
    unit = re.sub(r'\^\s*\{?([23])\}?', r'^\1', unit)
    return re.sub(r'[a-z]+', lambda m: _UNIT_ALIASES.get(m[0], m[0]), unit).replace(' ', '')


def _quantity(text):
    """从答案文本中分离数值部分和单位部分。

    返回 (数值文本, 单位规范形式)，无单位时单位为 None。
    度数和美元单独处理因为它们的标记符号不是普通单位词。
    """
    # 度数：匹配末尾的 ^\circ 或 °
    degree = re.search(r'(?:\^\s*(?:\\circ\b|\{\s*\\circ\s*\})|°)\s*$', text)
    if degree:
        return text[:degree.start()].strip(), 'degree'
    # 美元：\$ 前缀
    if text.startswith(r'\$'):
        return text[2:].strip(), 'dollar'
    # 普通单位：用 _UNIT 正则在末尾匹配单位词
    match = re.search(rf'(?<![A-Za-z\\])({_UNIT})\s*$', text, re.I)
    if match and text[:match.start()].strip():
        # 单字母单位 m/s 紧邻字母时可能是代数变量（如 5m 中的 m），不当单位处理
        if match[1].lower() in ('m', 's') and not text[match.start() - 1].isspace():
            return text, None
        value = text[:match.start()].strip()
        return value, _canonical_unit(match[1])
    return text, None


def _option_label(text, problem):
    """处理选择题：判断答案是否是一个选项标签（如 A、B、C、D）。

    从题干提取所有 (A) 形式的选项标签，少于 2 个则不是选择题。
    答案需匹配 (A) 或 A. 或 A 形式，且标签在题干标签集合内。
    排除包含多个标签的答案（others - {label} 不为空说明混了多个标签）。
    返回标签字母或 None。
    """
    labels = set(re.findall(r'\(([A-Z])\)', problem))
    if len(labels) < 2:
        return None
    # 匹配答案为 (A) 或 A. 或 A 的形式
    match = re.match(r'^\(([A-Z])\)|^([A-Z])(?:\s*[.。])?$', text)
    if match:
        label = match[1] or match[2]
        others = set(re.findall(r'\(([A-Z])\)', text))
        # 标签必须在题干选项中，且答案不能混入其他标签
        if label in labels and not (others - {label}):
            return label
    return None


def _parse_interval(text):
    """将区间表示法文本解析为 sympy.Interval 对象。

    处理：去掉前导变量声明（如 x \\in）→ 处理并集（\\cup 分割递归）
    → 解析端点和开闭区间（[ 闭端点，( 开端点）。
    无穷不能是闭端点，左端点必须小于右端点。
    """
    # 去掉前导的 "x \in" 形式的变量声明
    text = re.sub(r'^\s*(?:[A-Za-z]|\\[A-Za-z]+)\s*\\in\s*', '', text).strip()
    # 并集：按 \cup 分割后递归解析每段
    if r'\cup' in text:
        return sympy.Union(*(_parse_interval(part.strip()) for part in text.split(r'\cup')))
    if len(text) < 5 or text[0] not in '([' or text[-1] not in ')]':
        raise ValueError('Expected interval notation')
    endpoints = _tuple_parts('(' + text[1:-1] + ')')
    if endpoints is None or len(endpoints) != 2:
        raise ValueError('Expected two interval endpoints')
    left, right = (_parse(part) for part in endpoints)
    if left is sympy.oo or right is -sympy.oo:
        raise ValueError('Invalid infinite endpoint')
    # 无穷不能是闭端点：[∞, ...] 或 [..., ∞] 无意义
    if (left == -sympy.oo and text[0] == '[') or (right == sympy.oo and text[-1] == ']'):
        raise ValueError('Infinity cannot be a closed endpoint')
    # 左端点必须小于右端点
    if sympy.simplify(left > right) is sympy.true:
        raise ValueError('Reversed interval endpoints')
    return sympy.Interval(left, right, left_open=text[0] == '(', right_open=text[-1] == ')')


def _inequality_interval(text):
    """将不等式链（如 a < x < b）转换为 sympy.Interval 对象。

    为什么需要这个函数？
      有些答案用不等式链表示解集，需要转成区间才能和区间形式的标准答案比较。

    处理：\\le/\\ge 转为 <=/>= → 匹配 "左值 op1 变量 op2 右值" 模式
    → 验证中间是单个变量 → > 方向翻转成 < 方向 → 检查单调性和端点顺序。
    返回 sympy.Interval 或 None（不是不等式链）。
    """
    # 将 LaTeX 不等号转为 ASCII 形式
    text = re.sub(r'\\leq?(?![A-Za-z])', '<=', text)
    text = re.sub(r'\\geq?(?![A-Za-z])', '>=', text)
    # 匹配 "左值 op1 变量 op2 右值" 的结构
    match = re.fullmatch(
        r'\s*(.+?)\s*(<=|>=|<|>)\s*([A-Za-z]|\\[A-Za-z]+)\s*(<=|>=|<|>)\s*(.+?)\s*', text)
    if not match:
        return None
    left, op1, variable, op2, right = match.groups()
    # 中间必须是单个符号变量，如 a < x < b 中的 x
    if not isinstance(_parse(variable), sympy.Symbol):
        return None
    a, b = _parse(left), _parse(right)
    if a.free_symbols or b.free_symbols:
        raise ValueError('Symbolic interval bounds are not supported')
    # > 方向翻转成 < 方向：a > x > b 等价于 b < x < a
    if op1.startswith('>') and op2.startswith('>'):
        a, b, op1, op2 = b, a, op2.replace('>', '<'), op1.replace('>', '<')
    # 非单调链（如 a < x > b）不能表示区间
    if not (op1.startswith('<') and op2.startswith('<')):
        raise ValueError('Not a monotone bounded inequality')
    # 左端点必须小于右端点
    if (a > b) is sympy.true:
        raise ValueError('Reversed inequality bounds')
    return sympy.Interval(a, b, left_open=op1 == '<', right_open=op2 == '<')


def _grade(prediction: str, gold: str, problem: str = ''):
    """判分核心：给定模型答案、标准答案、题干，返回 {correct, reason, ...}。

    流程：
      1. 先把两个答案都 normalize_latex 统一写法；
      2. 空答案直接判错；
      3. 根据题干内容做特殊处理：
         - 题干问"百分之多少"时，去掉答案末尾的 % 再比（因为标准答案常省略 %）；
         - 题干问"所有解/所有整数"时，把 "1,2,3" 当成无序集合 {1,2,3} 而不是元组；
         - 答案是纯文本（人名、单词等）时直接字符串比，避免被 sympy 当成字母乘积；
      4. 把两个答案都 _parse 成 sympy 对象，用 _equivalent 比较；
      5. 如果解析或比较出错，退化为字符串相等比较（总比报错好）。

    返回的 reason 字段记录"为什么对/为什么错"，方便后续统计。
    """
    normalized_prediction = normalize_latex(prediction)
    normalized_gold = normalize_latex(gold)
    result = {'correct': False, 'reason': 'empty_answer'}
    if not normalized_prediction or not normalized_gold:
        return result
    rules = []
    # 从题干提取被 "what is/compute/find/determine" 等短语引用的 $...$ 表达式。
    # 模型可能把题干要求计算的表达式作为答案左值（如题干问 "what is $x+y$"，答案写 "x+y=5"），
    # 此时需要去掉左边的表达式只保留右边的值。
    requested = re.findall(r'\b(?:what is|compute|find|determine|express the quotient)\s*\$([^$]+)\$', problem, re.I)
    # 将表达式转为可比较的规范键：去掉花括号和空格后归一化。
    # 定义在内部是因为只在 drop_requested_label 中使用，且依赖外部的 normalize_latex。
    def expression_key(text):
        return re.sub(r'[{}\s]', '', normalize_latex(text)).rstrip('.')
    # 检查答案是否为"表达式 = 值"且表达式恰好是题干要求计算的式子，如果是则去掉左值只保留值。
    # 定义在内部是因为需要访问外部的 requested 列表、expression_key 函数和 rules 列表。
    def drop_requested_label(text):
        if text.count('=') == 1:
            left, right = text.split('=')
            if any(expression_key(left) == expression_key(target) for target in requested):
                rules.append('requested_expression_label')
                return right.strip()
        return text
    normalized_prediction = drop_requested_label(normalized_prediction)
    normalized_gold = drop_requested_label(normalized_gold)
    option_prediction = _option_label(normalized_prediction, problem)
    option_gold = _option_label(normalized_gold, problem)
    if option_prediction is not None or option_gold is not None:
        return {'correct': option_prediction is not None and option_prediction == option_gold,
                'reason': 'option_label_comparison'}

    prediction_value, prediction_unit = _quantity(normalized_prediction)
    gold_value, gold_unit = _quantity(normalized_gold)
    if prediction_unit and gold_unit and prediction_unit != gold_unit:
        return {'correct': False, 'reason': 'unit_mismatch'}
    # Explicit target units establish the answer convention; otherwise require
    # the question to explicitly request the unit. No cross-unit conversion.
    # 从题干 "in/how many + 单位" 提取要求的单位
    requested_units = {
        _canonical_unit(m[1]) for m in re.finditer(
            rf'\b(?:in|how many)\s+({_UNIT})(?![A-Za-z])', problem, re.I)}
    if re.search(r'\b(?:in|how many)\s+degrees\b', problem, re.I):
        requested_units.add('degree')
    # A length question can inherit its sole explicit length unit from the
    # numeric measurements in its statement. Mixed units remain ambiguous.
    if not requested_units and re.search(r'\bhow long\b', problem, re.I):
        stated = {_canonical_unit(m[1]) for m in re.finditer(
            rf'\d\s*({_UNIT})(?![A-Za-z])', problem, re.I)}
        length_units = {'inch', 'foot', 'cm', 'mm', 'km', 'm'}
        if len(stated) == 1 and stated <= length_units:
            requested_units.update(stated)
    if gold_unit or (prediction_unit and prediction_unit in requested_units):
        normalized_prediction, normalized_gold = prediction_value, gold_value
        rules.append('answer_unit_annotation')
    elif prediction_unit:
        return {'correct': False, 'reason': 'unconfirmed_unit'}
    if re.search(r'\bgrade\b', problem, re.I):
        # 去掉年级答案的序数后缀：如 "5th grade" → "5"、"5^{th} grade" → "5"
        # 只在题干含 "grade" 时使用
        def grade_number(text):
            return re.sub(r'^(\d+)\s*(?:\^\{(?:st|nd|rd|th)\}|st|nd|rd|th)?\s+grade$', r'\1', text)
        normalized_prediction = grade_number(normalized_prediction)
        normalized_gold = grade_number(normalized_gold)

    # 判断题干是否要求无序答案：匹配 "all the ... integers/solutions/roots/values" 等短语
    unordered = bool(re.search(
        r'\ball\s+(?:the\s+)?(?:such\s+)?(?:(?:integer|real)\s+)?(?:integers|solutions|roots|values)\b', problem, re.I))
    unordered = unordered or bool(re.search(r'\b(?:find|enter)\s+the\s+(?:real\s+)?roots\b', problem, re.I))
    if not unordered:
        # 去掉数字的千位分隔符：如 "1,000,000" → "1000000"、"1\,000\,000" → "1000000"
        # 只在非无序答案时处理：无序答案中的逗号是元素分隔符不是千分位
        # 定义在内部是因为需要访问外部的 raw（原始未归一化文本）判断是否是千分位格式
        def thousands(text, raw):
            if re.fullmatch(r'[+-]?\d{1,3}(?:\s*\\,\s*\d{3})+(?:\.\d+)?', raw.strip()):
                return re.sub(r'\s+', '', text)
            if re.fullmatch(r'[+-]?\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?', text):
                return re.sub(r'[,\s]', '', text)
            return text
        normalized_prediction = thousands(normalized_prediction, prediction)
        normalized_gold = thousands(normalized_gold, gold)
    # A percentage-valued question commonly omits the percent sign in its target.
    # 题干问百分比时，去掉答案末尾的 % 再比（标准答案常省略 %）
    if re.search(r'\b(?:what|which)\s+(?:percent|percentage)\b|\bas a percentage\b', problem, re.I):
        normalized_prediction = re.sub(r'(?:\\%|%)$', '', normalized_prediction).strip()
        normalized_gold = re.sub(r'(?:\\%|%)$', '', normalized_gold).strip()
        rules.append('percentage_answer_context')

    # 结合题干上下文将答案解析为 sympy 对象，处理 _parse 无法覆盖的上下文相关情况。
    # 定义在内部是因为需要访问外部的 problem、unordered、rules 等上下文变量。
    def parse_in_context(text, raw):
        # 先尝试将不等式链转为区间
        bounded_interval = _inequality_interval(text)
        if bounded_interval is not None:
            return bounded_interval
        # Only explicitly typeset prose may annotate a value. Bare 'x y'
        # is an algebraic product, not a descriptive answer label.
        # 处理 \text{...}= 形式的文本标签赋值
        label = re.match(r'^\s*\\text\{([A-Za-z]+(?:\s+[A-Za-z]+)+)\}\s*=', raw)
        if label and text.count('=') == 1:
            left, right = text.split('=')
            if left.strip() == label[1]:
                return Assignment(left.strip().casefold(), _parse(right))
        # 根据题干上下文或语法判断是否为区间
        interval_context = bool(re.search(r'\b(?:interval|domain|range)\b|set of all possible values', problem, re.I))
        interval_syntax = r'\infty' in text or r'\cup' in text or r'\in ' in text or text.startswith('[') or text.endswith(']')
        if (interval_context or interval_syntax) and (text.startswith(('(', '[')) or r'\in ' in text):
            return _parse_interval(text)
        # 无序答案的 \pm 展开：a\pm b → {a+b, a-b}
        if unordered and r'\pm' in text and not text.startswith(('(', '[', r'\{')):
            if text.count(r'\pm') != 1:
                raise ValueError('Ambiguous multiple plus-minus signs')
            return sympy.FiniteSet(_parse(text.replace(r'\pm', '+')),
                                   _parse(text.replace(r'\pm', '-')))
        # 无序答案的逗号列表转无序集合（括号包裹的元组保持有序）
        if unordered:
            # Bare comma lists can be unordered solutions; bracketed tuples stay ordered.
            if not text.startswith(('(', '[', r'\{')):
                parts = _tuple_parts('(' + text + ')')
                if parts:
                    return sympy.FiniteSet(*(_parse(part) for part in parts))
        # 兜底：以上都不是，交给 _parse 处理
        return _parse(text)
    # Explicit text, names, and prose must not become commutative letter products.
    # 判断答案是否为纯文本（人名、单词等），而非数学表达式。
    # 纯文本不能被 sympy 解析（会被当成字母乘积），需要直接字符串比较。
    # 定义在内部是因为只在当前上下文中使用，逻辑上属于 _grade 的一部分。
    def textual(raw, normalized):
        if r'\text' in raw:
            # 含 \text 命令：检查内容是否为纯字母短语
            return bool(re.fullmatch(r"[A-Za-z]+(?:[ '-][A-Za-z]+)*", normalized))
        # 不含 \text：检查是否为多字母单词或单个大写字母
        # 多字母要求至少 2 个字母，避免把 pi、in 等数学常数/关键词误判为文本
        # 单字母要求大写，小写单字母通常是数学变量（如 x）
        return (bool(re.fullmatch(r"[A-Za-z]{2,}(?:[ '-][A-Za-z]{2,})*", normalized))
                or bool(re.fullmatch(r'[A-Z]', normalized)))
    if textual(prediction, normalized_prediction) or textual(gold, normalized_gold):
        explicit_text = r'\text' in prediction or r'\text' in gold
        word_answer = len(normalized_prediction) > 1 and len(normalized_gold) > 1
        a, b = normalized_prediction, normalized_gold
        if explicit_text or word_answer:
            a, b = a.casefold(), b.casefold()
        return {'correct': a == b, 'reason': 'text_comparison'}
    try:
        parsed_prediction = parse_in_context(normalized_prediction, prediction)
        parsed_gold = parse_in_context(normalized_gold, gold)
        if re.search(r'\bvector\b', problem, re.I):
            # Coordinate notation may denote a column vector in a vector task;
            # row matrices and general ordered tuples retain their structure.
            if isinstance(parsed_prediction, sympy.Tuple) and isinstance(parsed_gold, sympy.MatrixBase) and parsed_gold.cols == 1:
                parsed_prediction = sympy.ImmutableMatrix(list(parsed_prediction))
                rules.append('vector_coordinate_notation')
            elif isinstance(parsed_gold, sympy.Tuple) and isinstance(parsed_prediction, sympy.MatrixBase) and parsed_prediction.cols == 1:
                parsed_gold = sympy.ImmutableMatrix(list(parsed_gold))
                rules.append('vector_coordinate_notation')
        if isinstance(parsed_prediction, sympy.FiniteSet) or isinstance(parsed_gold, sympy.FiniteSet):
            rules.append('finite_set_comparison')
        return {'correct': bool(_equivalent(parsed_prediction, parsed_gold)),
                'reason': 'symbolic_comparison', 'context_rules': rules}
    except Exception as error:
        # Preserve token boundaries, including whitespace between prose words.
        return {'correct': normalized_prediction == normalized_gold,
                'reason': 'string_fallback', 'error': type(error).__name__}


def grade_answer(prediction: str, gold: str, timeout_seconds: float = 5.0, *, problem: str = ''):
    """对外接口：判分，但放在子进程里跑，超时就返回"超时"。

    为什么要开子进程？
      _grade 里会调 sympy.simplify，这个函数有时候会卡很久（甚至死循环）。
      如果直接在主进程跑，一道题卡住整个评测就停了。
      放到子进程里，超时可以杀掉，主流程继续跑下一题。

    返回值和 _grade 一样是 {correct, reason, ...}，
    但多了 timeout、worker_error 这两种失败原因。
    """
    if prediction is None or gold is None:
        return {'correct': False, 'reason': 'missing_answer'}
    if timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive')
    try:
        completed = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
            input=json.dumps([prediction, gold, problem]), text=True, capture_output=True,
            timeout=timeout_seconds, check=True,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {'correct': False, 'reason': 'timeout'}
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return {'correct': False, 'reason': 'worker_error', 'error': type(error).__name__}


def is_equivalent(prediction: str, gold: str, *, problem: str = '') -> bool:
    """简化版接口：只返回 True/False，不要诊断信息。

    就是把 grade_answer 的结果取 ['correct']，给只想知道对错的调用方用。
    """
    return grade_answer(prediction, gold, problem=problem)['correct']


if __name__ == '__main__':
    # 这个脚本有两种用法：
    # 1. 被 grade_answer 用子进程调起来（带 --worker 参数）：
    #    从 stdin 读 [prediction, gold, problem]，算完把结果 JSON 打到 stdout；
    # 2. 直接 python grader.py 运行：跑下面的内置测试用例，
    #    验证各种 LaTeX 写法是否都能正确判等，相当于自测。
    if '--worker' in sys.argv:
        print(json.dumps(_grade(*json.load(sys.stdin))))
        sys.exit(0)
    cases = [
        # 元组中 \tfrac 和 \left\right 的归一化
        (r'(3,\tfrac{\pi}{2})', r'\left(3,\frac{\pi}{2}\right)', True),
        # 纯文本答案（人名）的比较，含句末句号
        ('Evelyn.', r'\text{Evelyn}', True),
        # 度数标记的忽略
        ('90', r'90^\circ', True),
        # 不同数值判错
        ('42', '43', False),
        # General formatting and regression checks, not answer-specific rules.
        # 省略花括号的分数与小数等价
        (r'\frac12', '0.5', True),
        # \boxed 包裹和句号的去除
        (r'\boxed{\left(3,\tfrac{\pi}{2}\right)}.',
         r'(3,\frac{\pi}{2})', True),
        # \boxed + \text 与纯文本的比较
        (r'\boxed{\text{Alice}}', 'Alice.', True),
        # 带度数标记的答案与纯数值不等价
        (r'45^{\circ}.', '45', False),
        # Unicode 度数符号 ° 的处理
        ('30°', '30', False),
        # 间距命令 \, 和 \quad 的去除
        (r'\frac{1}{2}\,\quad', '0.5', True),
        # 小数点不被当作千分位分隔符
        ('1.2', '12', False),
        # 阶乘不被忽略
        ('5!', '5', False),
        # 元组有序性：顺序不同则不等价
        ('(1,2)', '(2,1)', False),
    ]
    for index, (prediction, gold, expected) in enumerate(cases, 1):
        actual = is_equivalent(prediction, gold)
        print(f'case{index}: {actual} (expected {expected})')
        assert actual == expected, f'case{index} failed'
    assert normalize_latex(r'\sin(90^\circ)') == r'\sin(90^\circ)'

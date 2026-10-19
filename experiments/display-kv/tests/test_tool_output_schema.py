"""Terminal schema errors, restricted regex and nullable parallel policy; CPU only."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = (Path(os.environ['TF_DISPLAY_SOURCE']) / 'src/tensorfold/cuda/tool_output.py'
          if os.environ.get('TF_TOOL_OUTPUT_INSTALLED') == '1'
          else Path(__file__).resolve().parents[1] / 'overlay/tool_output.py')
spec = importlib.util.spec_from_file_location('native_review_fixture', SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError('native guard module not loadable')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def parse(schema, value, **request):
    tools = [{'type':'function','function':{'name':'check','parameters':schema}}]
    text = '<tool_call>' + json.dumps({'name':'check','arguments':{'x':value}}) + '</tool_call>'
    return native.ToolOutputGuard(tools, request={'tools':tools, **request}).parse(text)


def field(spec):
    return {'type':'object','properties':{'x':spec},'required':['x']}


def recursive(spec):
    return {**field(spec), '$defs':{'loop':{'$ref':'#/$defs/loop'}}}


LOOP = {'$ref':'#/$defs/loop'}


class SchemaSafetyTests(unittest.TestCase):
    def rejected(self, schema, value):
        with self.assertRaises(native.ModelOutputError) as caught:
            parse(schema, value)
        self.assertEqual(caught.exception.error['type'], 'model_output_error')
        self.assertEqual(caught.exception.error['reason'], 'invalid_executable_tool_output')
        self.assertNotIn('SECRET_SENTINEL', str(caught.exception))

    def test_evaluation_recursive_not(self):
        self.rejected(recursive({'not':LOOP}), 1)

    def test_evaluation_recursive_if(self):
        self.rejected(recursive({'if':LOOP,'then':False,'else':True}), 1)

    def test_evaluation_recursive_then_else(self):
        for cond, branch in [(True,'then'),(False,'else')]:
            with self.subTest(branch=branch):
                self.rejected(recursive({'if':cond,branch:{'not':LOOP}}), 1)

    def test_evaluation_recursive_anyof(self):
        for branches in [[LOOP,True],[True,LOOP]]:
            self.rejected(recursive({'anyOf':branches}), 1)

    def test_evaluation_recursive_oneof(self):
        self.rejected(recursive({'oneOf':[True,LOOP]}), 1)

    def test_evaluation_recursive_allof(self):
        self.rejected(recursive({'allOf':[True,{'not':LOOP}]}), 1)

    def test_evaluation_recursive_contains(self):
        for spec in [{'contains':{'not':LOOP}}, {'contains':LOOP,'minContains':0}]:
            self.rejected(recursive(spec), [1])

    def test_evaluation_nested_arrays(self):
        schema = {**field({'not':{'$ref':'#/$defs/arrays'}}),
                  '$defs':{'arrays':{'type':'array','items':{'$ref':'#/$defs/arrays'}}}}
        value = []
        for depth in range(61):
            if depth in (1,20,40,60):
                with self.subTest(depth=depth): self.rejected(schema, value)
            value = [value]

    def test_evaluation_real_mismatch_combinators_still_work(self):
        for spec,value in [({'not':{'type':'string'}},1),
                           ({'if':{'type':'string'},'then':False,'else':True},1),
                           ({'anyOf':[{'type':'string'},{'type':'integer'}]},1),
                           ({'oneOf':[{'type':'string'},{'type':'integer'}]},1),
                           ({'allOf':[{'type':'integer'},{'minimum':1}]},1),
                           ({'contains':{'type':'integer'}},['x',1])]:
            self.assertTrue(parse(field(spec),value)[1])

    def test_evaluation_invalid_unused_schema_rejects(self):
        for spec in [{'not':{'unsupportedAssertion':True}},
                     {'if':True,'else':{'pattern':'(a+)+'}},
                     {'anyOf':[True,{'type':'invalid'}]},
                     {'contains':{'$ref':'#/missing'},'minContains':0}]:
            self.rejected(field(spec), [])
        # A local pointer into an annotation still names an actual schema.
        self.rejected({**field({'if':True,'else':{'$ref':'#/default'}}),
                       'default':{'pattern':'(a+)+'}}, 1)

    def test_evaluation_evaluator_budget_not_inverted(self):
        # Shallow schema/instance: evaluation count, NOT recursion depth.
        for spec in [{'not':{'items':True}},
                     {'if':{'items':True},'then':True,'else':True},
                     {'anyOf':[True,{'items':True}]},
                     {'oneOf':[True,{'allOf':[{'items':True},False]}]},
                     {'allOf':[{'not':{'items':True}}]},
                     {'contains':{'items':True},'minContains':0}]:
            value = [[0]*11000] if 'contains' in spec else [0]*11000
            with self.subTest(spec=spec): self.rejected(field(spec), value)

    def test_evaluation_budget_error_kind_and_shared_union_accounting(self):
        # Refs share small acyclic schemas, expanding evaluation work only.
        defs={'leaf':True}
        target={'$ref':'#/$defs/leaf'}
        for level in range(14):
            defs[str(level)]={'allOf':[target,target]}
            target={'$ref':'#/$defs/'+str(level)}
        for spec in [{'not':target}, {'if':target,'else':True},
                     {'anyOf':[True,target]}, {'oneOf':[True,target]},
                     {'allOf':[True,target]}, {'contains':target,'minContains':0}]:
            value=[1] if 'contains' in spec else 1
            schema={**field(spec),'$defs':defs}
            with self.subTest(spec=spec):
                self.rejected(schema,value)
                # Public errors deliberately hide internals; inspect private kind
                # here to exclude a mismatch/schema-depth/size false positive.
                if hasattr(native,'_EvaluationError'):
                    with self.assertRaises(native._EvaluationError) as caught:
                        native._schema({'x':value},schema)
                    self.assertIn('budget exhausted',str(caught.exception))

    def test_regex_pathological_and_multiple_variable_repeats(self):
        # Deliberately small even on the unsafe baseline: no hanging RED probe.
        for pattern in ['^(a+)+$', '^a+a+$', '^a*a*$', '^a{1,3}a+$', '^a?b*$', '^[a-z]+[0-9]+$']:
            with self.subTest(pattern=pattern): self.rejected(field({'pattern':pattern}), 'a'*12)

    def test_regex_unsupported_regex_assertions(self):
        for pattern in ['(ab)', 'a|b', r'^(a)\1$', '(?=a)a', '(?:a)', r'\bword\b', '(?i)a', '^a++$', '^a*?$', 'a+', '^a+', 'a+$', '^a{0,999999}$', '[', r'\x61']:
            with self.subTest(pattern=pattern): self.rejected(field({'pattern':pattern}), 'a')

    def test_regex_unsafe_patterns_never_reach_compiler(self):
        for pattern in ['^(a+)+$', '^a+a+$', r'^(a)\1$', '(?=a)']:
            with self.subTest(pattern=pattern), patch.object(native.re,'compile') as compile_regex:
                with self.assertRaises(native._EvaluationError):
                    native._safe_pattern(pattern, native._Budget())
                compile_regex.assert_not_called()

    def test_regex_patternproperties_and_unused_patterns(self):
        for pattern in ['^(a+)+$', '^a+a+$', '(?=a)']:
            self.rejected(field({'patternProperties':{pattern:True}}), {})
            self.rejected(field({'patternProperties':{pattern:True}}), {'a'*12:1})
            self.rejected(field({'properties':{'unused':{'pattern':pattern}}}), {})

    def test_regex_allowed_patterns(self):
        for pattern,value in [(r'^[a-z][a-z0-9_-]*$', 'file_9-name'),
                              (r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$', '2026-10-19'),
                              (r'^\d{4}-\d{2}-\d{2}$', '2026-10-19'),
                              ('^a+$','aaa'), ('^.*$','plain'), ('^a{2,3}$','aaa'),
                              ('ab[0-9]','xxab3yy'), (r'a\.b','xxa.bxx'),
                              ('^[^/]*$','filename'), ('^$','')]:
            with self.subTest(pattern=pattern): self.assertTrue(parse(field({'pattern':pattern}),value)[1])
        self.assertTrue(parse(field({'patternProperties':{'^[a-z][a-z0-9_-]*$':{'type':'integer'}},'additionalProperties':False}), {'key_1':1})[1])

    def test_regex_allowed_nonmatches_not_inverted_as_errors(self):
        self.assertTrue(parse(field({'not':{'pattern':'^[a-z]+$'}}),'123')[1])
        self.rejected(field({'pattern':'^[a-z][a-z0-9_-]*$'}),'9bad')
        self.rejected(field({'pattern':'^[0-9]{4}-[0-9]{2}-[0-9]{2}$'}),'26-10-19')

    def test_regex_input_pattern_and_work_bounds(self):
        self.rejected(field({'not':{'pattern':'^a*$'}}), 'a'*9000)
        self.rejected(field({'pattern':'a'*300}), 'a'*300)
        self.rejected(field({'not':{'patternProperties':{'a':True}}}), {'a'*9000:1})
        self.rejected(field({'not':{'pattern':'a{128}b{128}'}}), 'a'*8192)
        # Per-evaluation aggregate regex work, not just a per-search cap.
        self.rejected(field({'not':{'items':{'pattern':'^a{128}a*$'}}}), ['a'*2048]*100)
        self.assertTrue(parse(field({'pattern':'^a*$'}),'a'*8192)[1])

    def test_regex_bounds_are_terminal_not_just_mismatches(self):
        cases=[({'pattern':'^a*$'},'a'*8193,'input limit'),
               ({'pattern':'a'*257},'a','pattern limit'),
               ({'pattern':'a{128}b{128}'},'a'*8192,'work budget'),
               ({'items':{'pattern':'^a{128}a*$'}},['a'*2048]*100,'work budget')]
        for spec,value,reason in cases:
            with self.subTest(reason=reason), self.assertRaises(native._EvaluationError) as caught:
                native._schema({'x':value},field({'not':spec}))
            self.assertIn(reason,str(caught.exception))

    def test_nullable_parallel_default_and_explicit_false(self):
        for parallel in [None,True,False]:
            self.assertEqual(native.ToolOutputGuard([],request={'parallel_tool_calls':parallel}).parse('ordinary answer'), ('ordinary answer',None))
        tools=[{'type':'function','function':{'name':'check','parameters':{}}}]
        text='<tool_call>{"id":"one","name":"check","arguments":{}}</tool_call><tool_call>{"id":"two","name":"check","arguments":{}}</tool_call>'
        for parallel in [None,True]:
            self.assertEqual(len(native.ToolOutputGuard(tools,request={'parallel_tool_calls':parallel}).parse(text)[1]),2)
        with self.assertRaises(native.ModelOutputError): native.ToolOutputGuard(tools,request={'parallel_tool_calls':False}).parse(text)


if __name__ == '__main__': unittest.main(verbosity=2)

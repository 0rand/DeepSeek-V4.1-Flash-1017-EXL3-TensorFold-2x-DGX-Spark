"""CPU regressions for strict native executable output, never inference."""
import importlib.util
import json
import os
from pathlib import Path
import unittest

MODULE = (Path(os.environ['TF_DISPLAY_SOURCE']) / 'src/tensorfold/cuda/tool_output.py'
          if os.environ.get('TF_TOOL_OUTPUT_INSTALLED') == '1'
          else Path(__file__).resolve().parents[1] / 'overlay/tool_output.py')
spec = importlib.util.spec_from_file_location('tool_output_test', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ToolOutputGuard, ModelOutputError = module.ToolOutputGuard, module.ModelOutputError
TOOLS = [{'type':'function','function':{'name':'terminal','parameters':{'type':'object','properties':{'x':{'type':'integer'}},'required':['x'],'additionalProperties':False}}}]
GOOD = '<tool_call>{"id":"call_fixture_exact","name":"terminal","arguments":{"x":7}}</tool_call>'

class Native(unittest.TestCase):
    def guard(self, **kw):
        request = {'tools': TOOLS, **kw}
        effective = [] if kw.get('tool_choice') == 'none' else TOOLS
        return ToolOutputGuard(effective, request=request)

    def test_malformed_and_atomic(self):
        for bad in ['<tool_call>{"name":"terminal",', '<tool_call>{}</tool_call>', '<tool_call>{"name":"terminal","arguments":[]}</tool_call>', '<｜DSML｜ calls><｜DSML｜ invoke name="terminal">', '<｜DSML｜tool_calls>', '<tool_call><function=terminal><parameter=x>1</function></tool_call>', '<tool_call>{"name":"terminal","arguments":{"x":1,"x":2}}</tool_call>', '<tool_call>{"name":"terminal","arguments":{"x":NaN}}</tool_call>', '<tool_call>{"name":"unoffered","arguments":{}}</tool_call>']:
            with self.subTest(bad=bad), self.assertRaises(ModelOutputError):
                self.guard().parse(GOOD + bad)

    def test_schema_before_truncation(self):
        for args in [{}, {'x':'7'}, {'x':True}, {'x':7,'extra':False}, {'x':float('inf')}, None, False, []]:
            with self.subTest(args=args), self.assertRaises(ModelOutputError):
                self.guard().parse(GOOD + '<tool_call>' + json.dumps({'name':'terminal','arguments':args}) + '</tool_call>', max_calls=1)

    def test_required_none_parallel(self):
        for text, choice, extra in [('answer','required',{}), (GOOD,'none',{}), (GOOD+GOOD,'auto',{'parallel_tool_calls':False})]:
            with self.subTest(choice=choice), self.assertRaises(ModelOutputError):
                self.guard(tool_choice=choice, **extra).parse(text)

    def test_literal_and_ids(self):
        for literal in ['```\n'+GOOD+'\n```', '`'+GOOD+'`', "'"+GOOD+"'", '> '+GOOD, '```\n<｜DSML｜ calls><｜DSML｜ invoke name="missing">\n```']:
            content, calls = self.guard().parse(literal)
            self.assertEqual(content, literal)
            self.assertIsNone(calls)
        content, calls = self.guard().parse(GOOD)
        self.assertEqual(content, '')
        self.assertEqual(calls[0]['id'], 'call_fixture_exact')
        nested = '<tool_call>'+json.dumps({'name':'terminal','arguments':{'x':7},'id':'copy_exact'})+'</tool_call>'
        self.assertEqual(self.guard().parse(nested)[1][0]['id'], 'copy_exact')

    def test_dsml_and_xml(self):
        ds = '<｜DSML｜ calls><｜DSML｜ invoke name="terminal"><｜DSML｜ parameter name="x" string="false">7</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'
        xml = '<tool_call><function=terminal><parameter=x>7</parameter></function></tool_call>'
        for text in [ds, xml]:
            self.assertEqual(json.loads(self.guard().parse(text)[1][0]['function']['arguments']), {'x':7})
        for text in [ds.replace('>7<','>true<'), ds.replace('>7<','>{<'), xml.replace('>7<','>True<'), ds.replace('</｜DSML｜ invoke>', '<｜DSML｜ parameter name="x" string="false">8</｜DSML｜ parameter></｜DSML｜ invoke>')]:
            with self.assertRaises(ModelOutputError): self.guard().parse(text)

    def test_literal_boundaries_do_not_hide_following_attempts(self):
        for literal in ['```\n<｜DSML｜ calls><｜DSML｜ invoke name="example">\n```', '> <tool_call>{broken', '`<tool_call>{broken`', '\"<tool_call>{broken\"', "'<tool_call>{broken'"]:
            self.assertEqual(self.guard().parse(literal),(literal,None))
            with self.assertRaises(ModelOutputError):
                self.guard().parse(literal+'\n<tool_call>{broken')
        with self.assertRaises(ModelOutputError):
            self.guard().parse('\"'+GOOD+'\" <tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call> \"example\"')
        ds='<｜DSML｜tool_calls><｜DSML｜invoke name="terminal"><｜DSML｜parameter name="x" string="false">7</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
        self.assertEqual(json.loads(self.guard().parse(ds)[1][0]['function']['arguments']),{'x':7})
        for marker in ['<foo:tool_call>{}</foo:tool_call>', '<|tool_call>call:missing{}<tool_call|>']:
            with self.assertRaises(ModelOutputError):self.guard().parse(marker)

    def test_choice_and_explicit_id_types(self):
        for choice in [False,0,[],{},'', 'unknown',{'type':'function','function':{'name':'unoffered'}}]:
            with self.subTest(choice=choice),self.assertRaises(ModelOutputError):
                self.guard(tool_choice=choice).parse(GOOD)
        for value in [None,False,0,[],{}]:
            text='<tool_call>'+json.dumps({'name':'terminal','arguments':{'x':7},'id':value})+'</tool_call>'
            with self.assertRaises(ModelOutputError):self.guard().parse(text)
        with self.assertRaises(ModelOutputError):self.guard(parallel_tool_calls=0).parse(GOOD)

    def test_nested_schema_constraints(self):
        schema={'type':'object','properties':{'nested':{'type':'object','properties':{'n':{'type':'number','minimum':1,'maximum':3},'label':{'enum':['ok','yes']}},'required':['n','label'],'additionalProperties':False},'items':{'type':'array','items':{'type':'integer'},'minItems':1,'uniqueItems':True}},'required':['nested','items'],'additionalProperties':False}
        tools=[{'type':'function','function':{'name':'check','parameters':schema}}]
        def parse(args):return ToolOutputGuard(tools).parse('<tool_call>'+json.dumps({'name':'check','arguments':args})+'</tool_call>')
        self.assertTrue(parse({'nested':{'n':2,'label':'ok'},'items':[1,2]})[1])
        for args in [{'nested':{'n':True,'label':'ok'},'items':[1]}, {'nested':{'n':4,'label':'ok'},'items':[1]}, {'nested':{'n':2,'label':'bad'},'items':[1]}, {'nested':{'n':2,'label':'ok'},'items':[False]}, {'nested':{'n':2,'label':'ok'},'items':[1,1]}]:
            with self.assertRaises(ModelOutputError):parse(args)
        schema['properties']['nested']['properties']['n']['multipleOf']=0.1
        self.assertTrue(parse({'nested':{'n':1.3,'label':'ok'},'items':[1]})[1])
        with self.assertRaises(ModelOutputError):parse({'nested':{'n':1.30000000001,'label':'ok'},'items':[1]})
        schema['properties']['nested']['unsupportedValidation']=True
        with self.assertRaises(ModelOutputError):parse({'nested':{'n':2,'label':'ok'},'items':[1]})

    def test_schema_contract_wrong_falsy_and_nonfinite(self):
        def parse(schema):
            tools=[{'type':'function','function':{'name':'empty','parameters':schema}}]
            return ToolOutputGuard(tools).parse('<tool_call>{"name":"empty","arguments":{}}</tool_call>')
        for schema in [{},True,{'type':'object','properties':{},'required':[],'additionalProperties':False}]:
            self.assertTrue(parse(schema)[1])
        for schema in [False,None,0,[],{'type':False},{'type':'object','required':False},{'type':'object','minProperties':False},{'type':'object','maxProperties':float('inf')},{'type':'object','additionalProperties':None},{'type':'object','properties':{'unused':{'type':'number','minimum':float('nan')}}},{'type':'object','properties':{'unused':{'minLength':False}}}]:
            with self.subTest(schema=schema),self.assertRaises(ModelOutputError):parse(schema)

    def test_ids_names_and_limits_are_not_repaired(self):
        for ident in ('', None, False, 0, [], {}):
            text = '<tool_call>' + json.dumps({'name':'terminal','arguments':{'x':7},'id':ident}) + '</tool_call>'
            with self.subTest(ident=ident), self.assertRaises(ModelOutputError):
                self.guard().parse(text)
        with self.assertRaises(ModelOutputError):
            self.guard().parse(GOOD + GOOD)
        for text in (GOOD.replace('terminal', 'TERMINAL'), GOOD + GOOD.replace('call_fixture_exact','second')):
            with self.subTest(text=text), self.assertRaises(ModelOutputError):
                self.guard().parse(text, max_calls=1)
        calls = self.guard().parse(GOOD + GOOD.replace('call_fixture_exact','second'))[1]
        self.assertEqual([call['id'] for call in calls], ['call_fixture_exact','second'])
        generated = self.guard().parse(GOOD.replace('"id":"call_fixture_exact",',''))[1][0]['id']
        self.assertTrue(generated.startswith('call_'))

    def test_protocol_in_argument_is_data(self):
        tools = [{'type':'function','function':{'name':'echo','parameters':{'type':'object','properties':{'text':{'type':'string'}}}}}]
        value = '<tool_call>{broken}</tool_call><｜DSML｜ calls>'
        text = '<tool_call>'+json.dumps({'id':'nested_exact','name':'echo','arguments':{'text':value}})+'</tool_call>'
        calls = ToolOutputGuard(tools).parse(text)[1]
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['text'], value)
        self.assertEqual(calls[0]['id'],'nested_exact')

if __name__ == '__main__': unittest.main(verbosity=2)

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

    def test_fence_info_cannot_be_its_own_closing_fence(self):
        xml = '<tool_call><function=terminal><parameter=x>7</parameter></function></tool_call>'
        dsml = '<｜DSML｜tool_calls><｜DSML｜invoke name="terminal"><｜DSML｜parameter name="x" string="false">7</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
        bodies = (GOOD, xml, dsml, '<tool_call>{broken', '<tool_call>{"name":"missing","arguments":{}}</tool_call>')
        for opener in ('~~~ ~~~', '~~~ ~~~~', '~~~~ ~~~~', '~~~ ~~~ info'):
            closing = opener.split()[0]
            for body in bodies:
                for closed in (False, True):
                    literal = opener + '\n' + body + ('\n' + closing if closed else '')
                    with self.subTest(opener=opener, body=body, closed=closed):
                        self.assertEqual(self.guard().parse(literal), (literal, None))
                        if closed:
                            content, calls = self.guard().parse(literal + '\n' + GOOD)
                            self.assertEqual(content, literal + '\n')
                            self.assertEqual(calls[0]['id'], 'call_fixture_exact')
                            with self.assertRaises(ModelOutputError):
                                self.guard().parse(literal + '\n<tool_call>{broken')

    def test_blockquote_prose_is_literal_only_on_its_own_line(self):
        for prefix in ('> Example: ', '> > Quoted example: ', '  > Description '):
            for body in (GOOD, '<tool_call>{broken'):
                literal = prefix + body
                with self.subTest(prefix=prefix, body=body):
                    self.assertEqual(self.guard().parse(literal), (literal, None))
                    with self.assertRaises(ModelOutputError):
                        self.guard().parse(literal + '\n<tool_call>{broken')

    def test_quoted_lines_cannot_own_unquoted_inline_state(self):
        bads = ('<tool_call>{broken', '<tool_call>{"name":"missing","arguments":{}}</tool_call>', '<tool_call>{"name":"terminal","arguments":{"x":"wrong"}}</tool_call>')
        for bad in bads:
            cases = []
            for prefix in ('> ', '> Example: ', '  > > Quoted: '):
                for run in ('`', '``'):
                    cases.append(prefix + run + '<tool_call>{broken\n' + bad + '\n' + run)
            cases.extend(('`\n' + bad + '\n> closing `', '`\n' + bad + '\n> quoted prose\n`', '`\n' + bad + '\n\n`'))
            for text in cases:
                for valid_prefix in ('', GOOD + '\n'):
                    with self.subTest(text=text, valid_prefix=bool(valid_prefix)), self.assertRaises(ModelOutputError):
                        self.guard().parse(valid_prefix + text)
        for literal in ('`\n' + GOOD + '\n`', '`> ' + GOOD + '`', 'before `> ' + GOOD + '`'):
            self.assertEqual(self.guard().parse(literal), (literal, None))

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

# Shared by portable parser and actual-installed JSON/SSE tests.
TEXT_TOOLS = [{'type':'function','function':{'name':'task','parameters':{
    'type':'object','properties':{'text':{'type':'string'}},
    'required':['text'],'additionalProperties':False}}}]


def text_call(value, form='json', ident='r1_exact'):
    payload = {'id':ident,'name':'task','arguments':{'text':value}}
    if form == 'nested':
        payload = {'id':ident,'type':'function','function':{
            'name':'task','arguments':{'text':value}}}
    elif form == 'encoded':
        payload['arguments'] = json.dumps(payload['arguments'])
    if form in ('json','nested','encoded'):
        return '<tool_call>'+json.dumps(payload)+'</tool_call>'
    if form == 'xml':
        return '<tool_call><function=task><parameter=text>'+value+'</parameter></function></tool_call>'
    if form == 'glm':
        return '<tool_call>task<arg_key>text</arg_key><arg_value>'+value+'</arg_value></tool_call>'
    gap = ' ' if form == 'dsml' else ''
    def tag(name): return '｜DSML｜'+gap+name
    outer = 'calls' if gap else 'tool_calls'
    return ('<'+tag(outer)+'><'+tag('invoke')+' name="task"><'+tag('parameter')+
            ' name="text" string="true">'+value+'</'+tag('parameter')+'></'+
            tag('invoke')+'></'+tag(outer)+'>')


def contamination_cases():
    suffixes = [('<tool_call>{broken', 'malformed'),
                ('<tool_call>{"name":"unoffered","arguments":{"secret":1}}</tool_call>', 'unavailable'),
                ('<tool_call>{"name":"task","arguments":{"text":7}}</tool_call>', 'schema')]
    # Preserve the independent review's exact original reproducer, without an ID.
    original = '<tool_call>{"name":"task","arguments":{"text":"`"}}</tool_call>'
    for suffix, failure in suffixes:
        yield ('original', '`', failure), original+'\n'+suffix+'\n`'
    for form in ('json','nested','encoded','xml','glm','dsml','dsml4'):
        for value, closer in [('`','`'), ('``','``'), ('```','```'),
                              ('line\n```\nend','\n```'), ('line\n~~~\nend','\n~~~')]:
            for suffix, failure in suffixes:
                yield (form, value, failure), text_call(value,form)+'\n'+suffix+'\n'+closer


def literal_controls():
    bad = '<tool_call>{broken'
    return ['`'+bad+'`', '``'+bad+'``', '```\n'+bad+'\n```',
            '~~~\n'+bad+'\n~~~', '```\n'+bad, '~~~\n'+bad,
            '> '+bad, "'"+bad+"'", '"'+bad+'"',
            '````\n`'+bad+'`\n```\nstill literal',
            '~~~\n```\n'+bad+'\n~~~',
            'intro `\n~~~\n`\n'+bad+'\n~~~',
            'intro `\n```\n`\n'+bad]


class ArgumentLiteralIsolation(unittest.TestCase):
    def test_atomic_contamination_regressions(self):
        for label, text in contamination_cases():
            with self.subTest(label=label):
                with self.assertRaises(ModelOutputError) as caught:
                    ToolOutputGuard(TEXT_TOOLS).parse(text)
                self.assertEqual(caught.exception.error['reason'],'invalid_executable_tool_output')

    def test_cross_envelope_arguments_are_not_markdown(self):
        for form in ('json','nested','encoded','xml','glm','dsml','dsml4'):
            for value in ('`','``','```','line\n```\nend','line\n~~~\nend'):
                with self.subTest(form=form,value=value):
                    text = text_call(value,form)+'\n'+text_call(value,form,'r1_second')
                    content, calls = ToolOutputGuard(TEXT_TOOLS).parse(text)
                    self.assertEqual(content,'\n')
                    self.assertEqual(len(calls),2)
                    for call in calls:
                        self.assertEqual(json.loads(call['function']['arguments']),{'text':value})
                    if form in ('json','nested','encoded'):
                        self.assertEqual([c['id'] for c in calls],['r1_exact','r1_second'])

    def test_literal_controls_and_nested_markers(self):
        for literal in literal_controls():
            with self.subTest(literal=literal):
                self.assertEqual(ToolOutputGuard(TEXT_TOOLS).parse(literal),(literal,None))
                content, calls = ToolOutputGuard(TEXT_TOOLS).parse(text_call('`')+'\n'+literal)
                self.assertEqual(content,'\n'+literal)
                self.assertEqual(calls[0]['id'],'r1_exact')
        value = '`<tool_call>{broken}</tool_call><｜DSML｜ calls>~~~'
        content, calls = ToolOutputGuard(TEXT_TOOLS).parse(text_call(value))
        self.assertEqual(content,'')
        self.assertEqual(json.loads(calls[0]['function']['arguments']),{'text':value})
        self.assertEqual(calls[0]['id'],'r1_exact')


    def test_quote_state_cannot_originate_in_arguments(self):
        for value in ("'", '"', "'`", '"~~~'):
            literal = "'<tool_call>{broken'"
            content, calls = ToolOutputGuard(TEXT_TOOLS).parse(text_call(value)+' '+literal)
            self.assertEqual(content,' '+literal)
            self.assertEqual(json.loads(calls[0]['function']['arguments']),{'text':value})
            with self.assertRaises(ModelOutputError):
                ToolOutputGuard(TEXT_TOOLS).parse(text_call(value)+" '<tool_call>{broken")


if __name__ == '__main__': unittest.main(verbosity=2)

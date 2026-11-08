"""Source headings must stay attached to claims taken from their tables."""
import deep_research as deep


def evidence(context=True):
    item = {'evidence_id': 'E1', 'path': 'rules.md', 'excerpt': '| 対象 | 設備の点検記録を五年間保存する。 |'}
    if context:
        item['context'] = [{'path': 'rules.md', 'text': '## 試験設備専用の手続', 'start_line': 1, 'end_line': 1}]
    return [item]


CLAIM = '設備の点検記録を五年間保存する。[E1]'


def test_table_scope_omission_enters_rewrite_and_repair_path():
    rows = evidence()
    record = {'attempts': []}
    _, issues = deep._check_final_candidate(CLAIM, rows, record)
    assert issues['limitation_lines']
    findings = deep._limitation_findings(CLAIM, rows)
    repaired, notes = deep._repair_limitations(CLAIM, findings)
    assert '試験設備専用の手続' in repaired
    assert notes and deep._validate_final_answer(repaired, {'E1'})
    assert deep._limitation_issues(repaired, rows) == []


def test_scope_on_same_claim_or_its_current_heading_is_retained():
    for answer in ['試験設備専用の手続：' + CLAIM, '## 試験設備専用の手続\n' + CLAIM]:
        assert deep._limitation_issues(answer, evidence()) == []


def test_scope_in_another_section_does_not_cover_this_claim():
    answer = '## 試験設備専用の手続\n別の説明。[E1]\n## 通常設備の手続\n' + CLAIM
    assert deep._limitation_issues(answer, evidence())


def test_heading_inside_excerpt_scopes_following_table():
    rows = evidence(False)
    rows[0]['excerpt'] = '## 試験設備専用の手続\n' + rows[0]['excerpt']
    assert deep._limitation_issues(CLAIM, rows)


def test_absent_or_other_document_heading_is_not_invented():
    assert deep._limitation_issues(CLAIM, evidence(False)) == []
    rows = evidence()
    rows[0]['context'][0]['path'] = 'other.md'
    assert deep._limitation_issues(CLAIM, rows) == []


def test_non_table_paragraph_does_not_gain_table_scope():
    rows = evidence()
    rows[0]['excerpt'] = '設備の点検記録を五年間保存する。'
    assert deep._limitation_issues(CLAIM, rows) == []


def test_following_source_heading_replaces_previous_table_scope():
    rows = evidence(False)
    rows[0]['excerpt'] = '## 試験設備専用の手続\n| 対象 | 点検を毎月行う。 |\n## 通常設備の手続\n| 対象 | 設備の点検記録を五年間保存する。 |'
    assert deep._limitation_issues('通常設備の手続：' + CLAIM, rows) == []
    assert deep._limitation_issues('試験設備専用の手続：' + CLAIM, rows)


def test_existing_explicit_scope_lead_is_not_replaced_by_a_table_title():
    rows = evidence()
    rows[0]['excerpt'] = '| 前項第1号の規定にかかわらず、次の事項を定める。 |\n| 一 承認記録を五年間保存する。 |'
    item = next(r for r in deep._source_limits(rows) if '承認記録' in r['raw'])
    assert item['scope'][0] == '前項第1号の規定にかかわらず'
    assert not item['table_scope']


def test_line_support_replacement_keeps_the_table_scope(monkeypatch):
    monkeypatch.setattr(deep, '_json_stage', lambda *a, **k: deep.LINE_UNSUPPORTED)
    answer = '試験設備専用の手続：' + CLAIM
    result, _ = deep._check_lines_support(
        answer, evidence(), query='設備', model='fake', budget=deep.DeepBudget(300), record={}
    )
    assert '試験設備専用の手続' in result
    assert deep._limitation_issues(result, evidence()) == []


def test_scope_lost_after_line_checks_cannot_pass_final_gate(tmp_path, monkeypatch):
    from test_deep_remediation import backend, research
    backend(monkeypatch, answer='試験設備専用の手続：' + CLAIM)
    monkeypatch.setattr(deep, '_check_lines_support', lambda *a, **k: (CLAIM, []))
    result = research(tmp_path, '# 試験設備専用の手続\n| 対象 | 設備の点検記録を五年間保存する。 |')
    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert result.diagnostics['final_answer']['result'] == 'rejected_limitations'


def test_child_table_title_keeps_parent_contract_scope():
    rows = evidence()
    rows[0]['excerpt'] = '### 点検チェックリスト\n' + rows[0]['excerpt']
    assert deep._limitation_issues('点検チェックリスト：' + CLAIM, rows)
    good = '## 試験設備専用の手続\n### 点検チェックリスト\n' + CLAIM
    assert deep._limitation_issues(good, rows) == []
    repaired, _ = deep._repair_limitations(CLAIM, deep._limitation_findings(CLAIM, rows))
    assert '試験設備専用の手続' in repaired and '点検チェックリスト' in repaired
    assert deep._limitation_issues(repaired, rows) == []


def test_table_column_headers_are_not_claim_replacement_targets():
    rows = evidence(False)
    rows[0]['excerpt'] = '| 項目 | 内容 |\n| --- | --- |\n| 設備 | 設備の点検記録を五年間保存する。 |'
    parsed = deep._source_limits(rows)
    assert not any(r['raw'] in {'項目', '内容'} for r in parsed)
    assert deep._best_source_row(deep._normalize_for_match('項目の理解について審議する。'), {'E1'}, parsed) is None

import pytest
from contractllm.store.version_store import VersionStore
from contractllm.store.models import ContractDefinition


@pytest.fixture
def store(tmp_path):
    return VersionStore(db_path=tmp_path / "test.db")


def test_register_and_get_contract(store):
    definition = ContractDefinition(
        name="test_contract",
        version="v1",
        system_prompt="You are helpful.",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        provider="openai",
        model="gpt-4o-mini",
    )
    store.register_contract(definition)

    fetched = store.get_definition("test_contract", "v1")
    assert fetched is not None
    assert fetched.name == "test_contract"
    assert fetched.version == "v1"
    assert fetched.schema_hash == definition.schema_hash


def test_register_updates_existing(store):
    definition_v1 = ContractDefinition(
        name="test_contract",
        version="v1",
        system_prompt="You are helpful.",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        provider="openai",
        model="gpt-4o-mini",
    )
    store.register_contract(definition_v1)

    definition_v2 = ContractDefinition(
        name="test_contract",
        version="v2",
        system_prompt="You are more helpful.",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        provider="openai",
        model="gpt-4o-mini",
    )
    store.register_contract(definition_v2)

    contracts = store.list_all()
    assert len(contracts) == 2


def test_get_nonexistent_returns_none(store):
    result = store.get_definition("nonexistent", "v1")
    assert result is None


def test_schema_hash_changes_on_content_change(store):
    v1 = ContractDefinition(
        name="test",
        version="v1",
        system_prompt="You are helpful.",
        input_schema={},
        output_schema={},
        provider="openai",
        model="gpt-4o-mini",
    )
    store.register_contract(v1)

    v2 = ContractDefinition(
        name="test",
        version="v1",
        system_prompt="You are VERY helpful.",  # Changed
        input_schema={},
        output_schema={},
        provider="openai",
        model="gpt-4o-mini",
    )
    store.register_contract(v2)

    fetched = store.get_definition("test", "v1")
    assert fetched.schema_hash != v1.schema_hash

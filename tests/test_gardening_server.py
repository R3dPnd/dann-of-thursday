"""Unit tests for the gardening MCP server tools."""

from src.mcp_servers.gardening_server import (
    add_garden_note,
    list_harvests,
    list_plantings,
    log_planting,
    record_harvest,
    search_garden_log,
)


class TestLogPlanting:
    def test_log_and_list(self, tmp_dann_home):
        log_planting("roma tomatoes", location="raised bed 2")
        log_planting("basil", location="raised bed 2")

        result = list_plantings()
        assert "roma tomatoes" in result
        assert "basil" in result
        assert "raised bed 2" in result

    def test_list_filters_by_location(self, tmp_dann_home):
        log_planting("roma tomatoes", location="raised bed 2")
        log_planting("carrots", location="front yard")

        result = list_plantings(location="front yard")
        assert "carrots" in result
        assert "roma tomatoes" not in result

    def test_empty_list_message(self, tmp_dann_home):
        assert list_plantings() == "No plantings found."


class TestHarvests:
    def test_record_and_list(self, tmp_dann_home):
        record_harvest("roma tomatoes", amount="2 lbs", notes="first batch")
        result = list_harvests()
        assert "roma tomatoes" in result
        assert "2 lbs" in result

    def test_filters_by_plant(self, tmp_dann_home):
        record_harvest("roma tomatoes", amount="2 lbs")
        record_harvest("basil", amount="a handful")

        result = list_harvests(plant="basil")
        assert "basil" in result
        assert "roma tomatoes" not in result


class TestSearchGardenLog:
    def test_matches_across_plantings_harvests_notes(self, tmp_dann_home):
        log_planting("roma tomatoes", location="raised bed 2")
        record_harvest("roma tomatoes", amount="2 lbs")
        add_garden_note("yellowing leaves, cut back watering", plant="roma tomatoes")

        result = search_garden_log("tomato")
        assert "planting" in result
        assert "harvest" in result
        assert "note" in result

    def test_no_match_returns_message(self, tmp_dann_home):
        log_planting("basil", location="raised bed 2")
        result = search_garden_log("nonexistent-plant")
        assert "Nothing in the garden log" in result

r"""Exposing the worker on a network must be an act, never a default.

WHY (issue #49). The developer wants to reach the worker from another machine on
their own network and has accepted the exposure. The measured state before this:
the socket listened on `127.0.0.1:8080` only, and the `Radmin VPN` interface at
26.33.142.160 had nothing bound to it.

WHAT MAKES THIS WORTH A TEST RATHER THAN A COMMENT.

`--host 127.0.0.1` is the **only** access control this server has. It runs with
no API key and CORS `*` -- its own boot banner says so -- and
`middleware_validate_api_key` (`tools/server/server-http.cpp:208`) returns `true`
immediately when no key is configured, so no route is protected. Widening the
bind does not weaken one control among several; it removes the only one.

So the default is pinned here, in a file that fails loudly, rather than trusted
to stay put. A future edit that moves it exposes the worker on **every** boot
after it, with nobody having chosen that and nothing in the output saying so --
the same shape as the reasoning effort that ran at `xhigh` for the life of the
project because no one had set it (report 35).

WHAT IS NOT ASSERTED. That the exposed path works. A bind is verified by
connecting to it from another address, which pytest cannot do from here. What is
checked is that exposure is opt-in, that the flag exists, and that the launcher
tells the truth about the two things that are not ours: the missing firewall rule
and the absent API key.
"""
import os
import re

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q2kxl-mtp.ps1")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_profile_still_defaults_to_loopback():
    """The one that matters. Every measured row was taken on a server nothing
    off the machine could reach, and moving this default changes that for every
    future boot without anyone choosing it."""
    assert re.search(r'\$BindAddress\s*=\s*[\'"]127\.0\.0\.1[\'"]', read(PROFILE)), (
        "the profile's bind default is not 127.0.0.1")


def test_the_profile_takes_the_bind_address_as_a_parameter():
    p = read(PROFILE)
    assert "$BindAddress" in p
    assert "--host $BindAddress" in p, (
        "the profile still hardcodes its host; the parameter is decorative")


def test_the_launcher_does_not_expose_by_default():
    """No switch, no exposure. A launcher that binds wide unless told otherwise
    is the same defect wearing a friendlier face.

    Asserted on the ARGUMENT that reaches the profile, not on where the string
    "0.0.0.0" sits in the file. The first version checked position and went red
    when Show-ServerStatus was added -- that function READS whether the socket
    is 0.0.0.0, it does not bind it, so the test was failing on structure while
    the behaviour was correct. A test that moves with refactoring is measuring
    the file, not the program."""
    s = read(SERVE)
    m = re.search(r"if\s*\(\s*\$Lan\s*\)", s)
    assert m, "serve.ps1 has no branch on -Lan"
    branch_line = len(s[:m.start()].splitlines())

    # Matches the hashtable form as well as the old array one. The first
    # version looked for "-BindAddress" with its leading dash and went blind the
    # moment splatting moved to a hashtable, where the key has no dash -- a test
    # that stops seeing the thing it guards is worse than no test, because it
    # goes green.
    binds = [(i, ln) for i, ln in enumerate(s.splitlines())
             if "BindAddress" in ln and "0.0.0.0" in ln]
    assert binds, "nothing ever passes a wide bind to the profile"
    for i, ln in binds:
        guarded = i > branch_line or "$Lan" in ln
        assert guarded, (
            "a wide bind is passed with nothing conditioning it on -Lan: %r"
            % ln.strip())


def test_the_launcher_offers_the_switch():
    assert re.search(r"\[switch\]\s*\$Lan", read(SERVE))


def test_it_reports_the_firewall_rule_it_cannot_add():
    """There is no inbound rule for the port, both adapters are classified
    Public, and adding one needs elevation the agent does not have. Without this
    the bind succeeds and the connection times out, which reads as a model
    problem."""
    s = read(SERVE)
    assert "New-NetFirewallRule" in s, (
        "serve.ps1 does not print the command for the rule it cannot add")
    assert re.search(r"Get-NetFirewallPortFilter|Get-NetFirewallRule", s), (
        "serve.ps1 does not check whether the rule already exists")


def test_adding_the_rule_needs_its_own_switch():
    """The launcher may now add the rule, because the developer asked it to.
    It may not do so as a side effect of -Lan: a switch that binds wide and
    edits the firewall in one step means nobody ever chose the second thing."""
    s = read(SERVE)
    assert re.search(r"\[switch\]\s*\$AllowFirewall", s), (
        "no separate switch guards the firewall change")
    m = re.search(r"if\s*\(\s*\$AllowFirewall\s*\)", s)
    assert m, "New-NetFirewallRule is not gated on -AllowFirewall"


def test_the_rule_is_added_through_an_elevation_prompt():
    """The agent is not admin and must not try to become it quietly. -Verb RunAs
    puts a consent dialog in front of the developer, which is the mechanism that
    makes this an authorised change rather than a silent one."""
    s = read(SERVE)
    assert "RunAs" in s, (
        "the firewall change does not go through an elevation prompt")


def test_the_rule_is_scoped_to_the_networks_that_asked_for_it():
    """26.0.0.0/8 is Radmin VPN's range; LocalSubnet is the Wi-Fi LAN the
    developer added afterwards. Allowing every remote address when two named
    networks were asked for is wider than the request, and the request is the
    authorisation."""
    s = read(SERVE)
    assert "26.0.0.0/8" in s, "the rule is not scoped to the Radmin range"
    assert "LocalSubnet" in s, "the rule does not admit the local LAN"
    assert "-RemoteAddress" in s, "the rule does not restrict remote addresses"


def test_it_checks_the_scope_of_an_existing_rule_not_merely_its_existence():
    """The first version skipped the whole branch when ANY rule was present, so
    a rule created when only Radmin was wanted could never be widened for
    Wi-Fi -- and the launcher would report 'rule present' while the LAN still
    timed out. Existence is not the property that matters; scope is."""
    s = read(SERVE)
    assert "Get-NetFirewallAddressFilter" in s, (
        "nothing reads the remote-address scope of the rule that already exists")


def test_it_removes_every_rule_it_owns_not_only_the_exact_name():
    """Renaming the rule between versions orphaned the old one. The first
    release created `llama-server 8080 (Radmin)`; the next removed
    `llama-server 8080` and left the first sitting there, so two rules existed
    and Windows evaluated their union -- a stale, narrower rule looking
    authoritative next to the real one. Remove by PREFIX, so a rule this script
    made under any past name is cleaned up."""
    s = read(SERVE)
    assert "llama-server $Port*" in s, (
        "the cleanup matches an exact name, so a renamed rule is orphaned")


def test_it_says_what_localsubnet_follows():
    """LocalSubnet means whichever network the machine is on, and the Wi-Fi
    adapter is classified Public. Taking the laptop elsewhere carries the rule
    along. Stated once, where the developer is choosing."""
    s = read(SERVE)
    m = re.search(r"if\s*\(\s*\$AllowFirewall\s*\)", s)
    after = s[m.start():]
    assert re.search(r"whatever network|any network you join|follows", after), (
        "the launcher does not say that LocalSubnet follows the machine")


def test_it_verifies_the_rule_landed_instead_of_assuming_the_prompt_was_accepted():
    """A UAC dialog can be dismissed. Reporting success because a command was
    launched is the same class of mistake as reporting residency from a
    projection."""
    s = read(SERVE)
    m = re.search(r"if\s*\(\s*\$AllowFirewall\s*\)", s)
    after = s[m.start():]
    assert after.count("Get-NetFirewallRule") >= 1, (
        "nothing re-checks the rule after the elevation attempt")


def test_it_states_the_missing_authentication_at_the_moment_of_exposure():
    """Once, where the developer is looking, and not on every loopback boot."""
    s = read(SERVE)
    m = re.search(r"if\s*\(\s*\$Lan\s*\)", s)
    after = s[m.start():]
    assert re.search(r"API key|api-key", after), (
        "the exposed path does not mention that no API key is set")

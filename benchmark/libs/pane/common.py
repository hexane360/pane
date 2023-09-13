
from datetime import datetime
import enum
import typing as t

import pyperf

import pane
from benchmark.common import AbstractBenchmark


class IssueState(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class MilestoneState(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class IssueStateReason(enum.Enum):
    COMPLETED = "completed"
    REOPENED = "reopened"
    NOT_PLANNED = "not_planned"


class AuthorAssociation(enum.Enum):
    COLLABORATOR = "COLLABORATOR"
    CONTRIBUTOR = "CONTRIBUTOR"
    FIRST_TIMER = "FIRST_TIMER"
    FIRST_TIME_CONTRIBUTOR = "FIRST_TIME_CONTRIBUTOR"
    MANNEQUIN = "MANNEQUIN"
    MEMBER = "MEMBER"
    NONE = "NONE"
    OWNER = "OWNER"


class User(pane.PaneBase):
    login: str
    id: int
    node_id: str
    avatar_url: str
    gravatar_id: t.Optional[str]
    url: str
    html_url: str
    followers_url: str
    following_url: str
    gists_url: str
    starred_url: str
    subscriptions_url: str
    organizations_url: str
    repos_url: str
    events_url: str
    received_events_url: str
    type: str
    site_admin: bool
    name: t.Optional[str] = None
    email: t.Optional[str] = None
    starred_at: t.Optional[datetime] = None


class IssueLabel(pane.PaneBase):
    id: int
    node_id: str
    url: str
    name: str
    description: t.Optional[str]
    color: t.Optional[str]
    default: bool


class Milestone(pane.PaneBase):
    url: str
    html_url: str
    labels_url: str
    id: int
    node_id: str
    number: int
    title: str
    description: t.Optional[str]
    creator: t.Optional[User]
    open_issues: int
    closed_issues: int
    created_at: datetime
    updated_at: datetime
    closed_at: t.Optional[datetime]
    due_on: t.Optional[datetime]
    state: MilestoneState = MilestoneState.OPEN


class Reactions(pane.PaneBase):
    url: str
    total_count: int
    plus_one: int = pane.field(aliases=("+1",))
    minus_one: int = pane.field(aliases=("-1",))
    laugh: int
    confused: int
    heart: int
    hooray: int
    eyes: int
    rocket: int


class Issue(pane.PaneBase):
    id: int
    node_id: str
    url: str
    repository_url: str
    labels_url: str
    comments_url: str
    events_url: str
    html_url: str
    number: int
    state: IssueState
    state_reason: t.Optional[IssueStateReason]
    title: str
    body: t.Optional[str]
    user: t.Optional[User]
    labels: list[t.Union[IssueLabel, str]]
    assignee: t.Optional[User]
    assignees: t.Optional[list[User]]
    milestone: t.Optional[Milestone]
    locked: bool
    active_lock_reason: t.Optional[str]
    comments: int
    closed_at: t.Optional[datetime]
    created_at: datetime
    updated_at: datetime
    closed_by: t.Optional[User]
    author_association: AuthorAssociation
    draft: bool = False
    body_html: t.Optional[str] = None
    body_text: t.Optional[str] = None
    timeline_url: t.Optional[str] = None
    reactions: t.Optional[Reactions] = None


class Benchmark(AbstractBenchmark):
    LIBRARY = "pane"

    def warmup(self, data: t.Any) -> None:
        Issue.from_data(data).into_data()

    def run_loader(self, data: t.Any) -> pyperf.Benchmark:
        return self._bench_loader_func(Issue.from_data, data)

    def run_dumper(self, data: t.Any) -> pyperf.Benchmark:
        obj = Issue.from_data(data)
        return self._bench_dumper_func(obj.to_data)
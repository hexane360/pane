#!/usr/bin/env python3

from build.util import project_wheel_metadata

msg = project_wheel_metadata('.')
version = msg.get('version')
project = msg.get('name')

print(f'project = "{project}"')
print(f'version = "{version}"')
print(f'tag = "v{version}"')
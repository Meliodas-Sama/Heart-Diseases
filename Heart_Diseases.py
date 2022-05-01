# Copyright 2018-2021 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
from collections import OrderedDict

import streamlit as st
from streamlit.logger import get_logger
import Pages

LOGGER = get_logger(__name__)

# Dictionary of
# demo_name -> (demo_function, demo_description)
DEMOS = OrderedDict(
    [
        ("—", (Pages.intro, None)),
        (
            "Classification Page",
            (
                Pages.Classification,
                """
Here you can enter data and see the classification.
""",
            ),
        ),
        (
            "Evaluation Page",
            (
                Pages.Evaluation,
                """
Here you can choose a model and see it's Evaluation.
""",
            ),
        )

    ]
)


def run():
    demo_name = st.sidebar.selectbox("Choose a Page", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name][0]

    if demo_name == "—":
        show_code = False
        st.write("# Welcome to Our Heart Diseases Website! 👋")
    else:
        show_code = st.sidebar.checkbox("Show code", True)
        st.markdown("# %s" % demo_name)
        description = DEMOS[demo_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()

    demo()

    if show_code:
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


if __name__ == "__main__":
    run()

"""This file should be imported if and only if you want to run the UI locally."""

import base64
import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from bnet.constants import PROJECT_ROOT_PATH
from bnet.di import global_injector
from bnet.open_ai.extensions.context_filter import ContextFilter
from bnet.server.chat.chat_service import ChatService, CompletionGen
from bnet.server.chunks.chunks_service import Chunk, ChunksService
from bnet.server.ingest.ingest_service import IngestService
from bnet.server.recipes.summarize.summarize_service import SummarizeService
from bnet.settings.settings import settings
from bnet.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "bnet/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "BNET"

SOURCES_SEPARATOR = "\n"


file_name_map = {
    "- Procurement Guidelines.docx": "BNET Procurement Guidlines",
    "Final - Procurement Policy 002.pdf": "BNET Procurement Policy",
    "EL_Sijil_Enterprise Architecture_v0.135.docx": "EL Sijil Enterprise Architecture",
}

file_name_map1 = {
    "BNET Procurement Guidlines": "- Procurement Guidelines.docx",
    "BNET Procurement Policy": "Final - Procurement Policy 002.pdf",
    "EL Sijil Enterprise Architecture": "EL_Sijil_Enterprise Architecture_v0.135.docx",
}


class Modes(str, Enum):
    KNOWLEDGE_MANAGEMENT = "Knowledge Management"
    RISK_MANAGEMENT = "Risk Management"
    LEGAL = "Legal"


MODES: list[Modes] = [Modes.KNOWLEDGE_MANAGEMENT, Modes.RISK_MANAGEMENT, Modes.LEGAL]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return ""


@singleton
class BnetUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.KNOWLEDGE_MANAGEMENT
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

    def _chat(
        self, message: str, history: list[list[str]], mode: Modes, *_: Any
    ) -> Any:

        if (
            "summary" in message.lower()
            or "summarise" in message.lower()
            or "summarize" in message.lower()
        ):
            self._system_prompt = settings().ui.default_summarization_system_prompt

        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            yield full_response

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.KNOWLEDGE_MANAGEMENT:
                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)

            case Modes.RISK_MANAGEMENT:
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)

            case Modes.LEGAL:
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)

            case Modes.SUMMARIZE_MODE:
                # Summarize the given message, optionally using selected files
                context_filter = None
                if self._selected_filename:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.KNOWLEDGE_MANAGEMENT:
                p = settings().ui.default_query_system_prompt
            case Modes.RISK_MANAGEMENT:
                p = settings().ui.default_risk_management_system_prompt
            # For any other mode, clear the system prompt
            case Modes.LEGAL:
                p = settings().ui.default_legal_system_prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.KNOWLEDGE_MANAGEMENT:
                return "Get contextualized answers from selected files."
            case Modes.RISK_MANAGEMENT:
                return "Find relevant chunks of text in selected files."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    # def _set_explanatation_mode(self, explanation_mode: str) -> None:
    #     self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(
                placeholder=self._system_prompt,
                interactive=interactive,
                elem_classes=["mode-class"],
            ),
            # gr.update(value=self._explanation_mode,elem_classes=["mode-class"]),
        ]

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            try:
                mapped_name = file_name_map.__getitem__(file_name)
            except KeyError:
                mapped_name = file_name
            files.add(file_name_map.__getitem__(file_name))
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files(), elem_classes=["mode-class"]),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files", elem_classes=["mode-class"]),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files(), elem_classes=["mode-class"]),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files", elem_classes=["mode-class"]),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files", elem_classes=["mode-class"]),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        
        try:
            self._selected_filename = file_name_map1.__getitem__(select_data.value)
        except:
            self._selected_filename = select_data.value
        
        print("asdasfasf :  ", select_data.value)
        print("\n\n\nSelected file name : ", self._selected_filename)
        print("\n\n\nSelected file data : ", select_data.value)
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox(select_data.value, elem_classes=["mode-class"]),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(
                primary_hue="yellow",
                # secondary_hue="#FFC300",  # Optional: Add accent colors here
                # neutral_hue="#FFFFFF",   # White background for neutral elements
                # background_fill="#FFFFFF",  # Set overall background to white
                # text_color="#000000",       # Black text for better readability
            ),
            css=".logo { "
            "display:flex;"
            "background-color: #F4F4F4;"
            "height: 90px;"
            "border-radius: 8px;"
            "border: 2px solid #F4F4F4;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 70% }"
            ".contain { display: flex !important; flex-direction: column !important; background-color: #FFFFFF;}"
            "#component-0, #component-3, #component-10, #component-8 { height: 100% !important; background-color: #FFFFFF;}"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important; backgroud-color:#FFFFFF}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFC300; }"  # #FFC300 line for separators
            ".avatar-image { background-color: #FFC300; border-radius: 2px; }"  # #FFC300 background for avatar images
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; color: #FFC300; }"  # Footer text in #FFC300
            ".footer-zylon-link { display: flex; margin-left: 5px; text-decoration: auto; color: #FFC300; }"  # Footer link in #FFC300
            ".footer-zylon-link:hover { color: #FFFFFF; }"  # Lighter #FFC300 on hover
            ".footer-zylon-ico { height: 20px; margin-left: 5px; background-color: #FFFFFF; border-radius: 2px; }"  # White background for icon
            ".gradio-container { background-color: #FFFFFF !important; color: #FFC300 !important; }"  # White container with #FFC300 text
            ".gradio-row { border: 1px solid #FFC300; }"  # #FFC300 border for rows
            ".gradio-column { padding: 10px; background-color: #4dff4d; color: #FFC300; }"
            ".upload-container { background-color: #4dff4d; color:#FFC300;}"
            ".button-class {background-color: #F4F4F4 !important;color: #FFC300 !important;border: 2px solid #F4F4F4;border-radius: 5px;padding: 10px 15px;}"
            ".button-class:hover {background-color: #FFC300 !important; color: white !important;cursor: pointer;    }"
            ".mode-class {background-color: #F4F4F4 !important;color: #FFC300 !important;border: 2px solid #F4F4F4;border-radius: 5px;padding: 10px 15px;}"
            ".mode1-class {background-color: #F4F4F4 !important;color: #FFC300 !important;border: 2px solid #F4F4F4;border-radius: 5px;padding: 10px 15px;}"
            ".wrapper.svelte-nab2ao {background-color:F4F4F4 ;} "
            ".light .bubble-wrap.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin {background: var(--background-fill-secondary);}"
            ".gradio-container.gradio-container-4-44-0 .contain textarea.svelte-1f354aw.svelte-1f354aw { color: #FFC300 !important ;}"
            "root {--body-background-fill: #ffffff;--body-text-color: #000000; --color-accent-soft: #ffcccc;--background-fill-primary: #ff9999; /* Light red background */--background-fill-secondary: #f5f5f5;--border-color-accent: #ff0000; /* Red border color */--border-color-primary: #ff3333; /* Darker red */--link-text-color-active: #cc0000; /* Red active link */}"
            """
.dark .bubble-wrap.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin {
    background: #F4F4F4 !important; /* Directly set to bright yellow */
}
"""
            """
input.svelte-1f354aw.svelte-1f354aw, textarea.svelte-1f354aw.svelte-1f354aw { background: #8a929c !important; color: #FFC300 } 
"""
            """
input.svelte-1f354aw.svelte-1f354aw::placeholder, 
textarea.svelte-1f354aw.svelte-1f354aw::placeholder {
    color: white; /* Golden yellow for placeholder text */
    opacity: 1; /* Ensure full opacity */
}
"""
            """
span.svelte-1gfkn6j { background: #FFC300 !important; } 
label.selected.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j { background: #FFC300 !important; } 
label.svelte-1b6s6s { background: #FFC300 !important; } 
.primary.svelte-cmf5ev { background: #8a929c !important; } 
.secondary.svelte-cmf5ev:hover, .secondary[disabled].svelte-cmf5ev   { backgroud: #FFFFFF !important }
.primary.svelte-cmf5ev:hover { background-color: #FFC300 !important; color: white !important;cursor: pointer; }
.secondary[disabled].svelte-cmf5ev { backgroud: #FFFFFF !important }
button[disabled].svelte-cmf5ev, a.disabled.svelte-cmf5ev { backgroud: #FFFFFF !important }
"""
            """
.flex-wrap.user.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin { background-color: #8a929c !important; color: white !important; border: 2px solid  #F4F4F4 !important}
"""
            """
.message.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin { color: white !important; }
"""
            """
:not(.component-wrap).flex-wrap.bot.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin { background-color: #8a929c !important; color: white !important; border: 2px solid  #F4F4F4 !important}
"""
            """
gradio-app .gradio-container.gradio-container-4-44-0 .contain .avatar-image { background-color: #F4F4F4 !important }
"""
            """
.avatar-container.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin { border: 2px solid  #F4F4F4 !important}
"""
            """
.avatar-container.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin { width: 50px !important ; height: 50px !important ;}
"""
            """
.label.svelte-1oa6fve p.svelte-1oa6fve.svelte-1oa6fve { color: #FFC300 !important ;}
"""
            """
tr.svelte-1oa6fve th.svelte-1oa6fve.svelte-1oa6fve { background : #8a929c !important }
"""
            """
.row_odd.svelte-1oa6fve.svelte-1oa6fve.svelte-1oa6fve { background : #F4F4F4 !important ; color: #FFC300 !important }
"""
            """
.block.svelte-12cmxck { background : #F4F4F4 !important ;}
"""
            """
label.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j { background : #8a929c !important ;}
"""
            """
label.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j:hover {background-color: #FFC300 !important; color: white !important;cursor: pointer;    }"
"""
            """
.secondary.svelte-cmf5ev { background : #F4F4F4 !important ;} 
"""
            """
input.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j  { background: #F4F4F4 !important ; }
"""
            """
tbody.svelte-82jkx>tr:nth-child(2n) { background: #F4F4F4 !important ; color : #FFC300 !important }
"""
            """
table.svelte-82jkx.svelte-82jkx {
  height: 300px !important ; /* Fixed height */
  overflow-y: auto; /* Enable vertical scrolling */
  overflow-x: auto; /* Enable horizontal scrolling */

}
"""
            """
.table-wrap.svelte-1oa6fve.svelte-1oa6fve.svelte-1oa6fve {
  overflow: auto !important;
  height: 300px !important; /* Set a fixed height for the wrapper */
}
"""
            """
.footer { display: none !important;}
"""
            """
footer.svelte-1rjryqp.svelte-1rjryqp.svelte-1rjryqp { display: none !important; }
"""
            """
.eta-bar.svelte-au1olv.svelte-au1olv { background: #F4F4F4 !important;}
""",
            # White column with #FFC300 text,
        ) as blocks:
            with gr.Row():
                # gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div> <div style='font-size: 20px; font-weight: bold; color:#FFFFFF';>BNET</div>")
                gr.HTML(
                    f"""
      
            <div class='logo'/><img src={logo_svg} alt=PrivateGPT>
            <div style="font-size: 20px; font-weight: bold; color:#F4F4F4"; margin-left:20px; padding:10px >BN</div>

            <div style="font-size: 40px; font-weight: bold; color:#FFC300"; margin-left:20px; padding:10px >BNET</div>
            </div>

            
    
        """
                )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    default_mode = self._default_mode
                    mode = gr.Radio(
                        [mode.value for mode in MODES],
                        label="Mode",
                        value=default_mode,
                        elem_classes=["mode-class"],
                    )
                    # explanation_mode = gr.Textbox(
                    #     placeholder=self._get_default_mode_explanation(default_mode),
                    #     show_label=False,
                    #     max_lines=3,
                    #     interactive=False,
                    #     elem_classes=["mode-class"]
                    # )
                    gr.HTML("<div class='upload-container'>")

                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                        elem_classes=["button-class"],
                    )
                    gr.HTML("</div>")

                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                        render=False,  # Rende#FFC300 under the button
                        elem_classes=["mode-class"],
                        elem_id="ingested-files-list",
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected File", max_lines=1
                    )
                    delete_file_button = gr.components.Button(
                        "",
                        # size="sm",
                        # visible=settings().ui.delete_file_button_enabled,
                        # interactive=False,
                    )
                    # delete_files_button = gr.components.Button(
                    #     "⚠️ Delete ALL files",
                    #     size="sm",
                    #     visible=settings().ui.delete_all_files_button_enabled,
                    #     elem_classes=["button-class"]
                    # )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    # delete_files_button.click(
                    #     self._delete_all_files,
                    #     outputs=[
                    #         ingested_dataset,
                    #         delete_file_button,
                    #         deselect_file_button,
                    #         selected_text,
                    #     ],
                    # )
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    # When mode changes, set default system prompt, and other stuffs
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input],
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                    def get_model_label() -> str | None:
                        """Get model label from llm mode setting YAML.

                        Raises:
                            ValueError: If an invalid 'llm_mode' is encounte#FFC300.

                        Returns:
                            str: The corresponding model label.
                        """
                        # Get model label from llm mode setting YAML
                        # Labels: local, openai, openailike, sagemaker, mock, ollama
                        config_settings = settings()
                        if config_settings is None:
                            raise ValueError("Settings are not configu#FFC300.")

                        # Get llm_mode from settings
                        llm_mode = config_settings.llm.mode

                        # Mapping of 'llm_mode' to corresponding model labels
                        model_mapping = {
                            "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                            "openai": config_settings.openai.model,
                            "openailike": config_settings.openai.model,
                            "azopenai": config_settings.azopenai.llm_model,
                            "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                            "mock": llm_mode,
                            "ollama": config_settings.ollama.llm_model,
                            "gemini": config_settings.gemini.model,
                        }

                        if llm_mode not in model_mapping:
                            print(f"Invalid 'llm mode': {llm_mode}")
                            return None

                        return model_mapping[llm_mode]

                with gr.Column(scale=7, elem_id="col"):
                    # Determine the model label based on the value of BNET_PROFILES
                    model_label = get_model_label()
                    if model_label is not None:
                        label_text = f"PROTIVITI-BNET"
                    else:
                        label_text = f"LLM: {settings().llm.mode}"

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(
                                None,
                                AVATAR_BOT,
                            ),
                            elem_classes=["mode-class"],
                            # retry_btn=None,
                            # undo_btn="Delete Previous",
                            # clear_btn="Clear"
                        ),
                        additional_inputs=[mode, upload_button, system_prompt_input],
                    )

            # with gr.Row():
            #     avatar_byte = AVATAR_BOT.read_bytes()
            #     f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
            #     gr.HTML(
            #         f"<div class='footer'><a class='footer-zylon-link' href='https://zylon.ai/'>Maintained by Zylon <img class='footer-zylon-ico' src='{f_base64}' alt=Zylon></a></div>"
            #     )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(BnetUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)

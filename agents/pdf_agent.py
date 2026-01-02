"""PDF Agent: Loads and processes PDF documents."""
import os
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import ConversationState, PDFData, AgentType, DataCollectionResult
from agents.base_agent import BaseAgent


class PDFAgent(BaseAgent):
    """Agent that loads and processes PDF documents."""
    
    def __init__(self, llm=None, pdf_directory: str = "./pdfs", **kwargs):
        super().__init__(llm=llm, agent_type=AgentType.PDF_AGENT, **kwargs)
        self.pdf_directory = Path(pdf_directory)
        self.parser = PydanticOutputParser(pydantic_object=PDFData)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a PDF processing agent. You load PDF files and extract structured information from them.

When a user mentions a PDF file, load it and extract the content. Provide a summary and key information.

{format_instructions}"""),
            ("human", """User input: {user_input}

Extract information from the PDF if mentioned, or ask the user for the PDF file path."""),
        ])
    
    def load_pdf(self, file_path: str) -> Optional[PDFData]:
        """
        Load and extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDFData object with extracted content, or None if error
        """
        try:
            full_path = Path(file_path)
            if not full_path.exists():
                # Try relative to pdf_directory
                full_path = self.pdf_directory / file_path
            
            if not full_path.exists():
                return None
            
            reader = PdfReader(str(full_path))
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            metadata = reader.metadata or {}
            
            return PDFData(
                filename=full_path.name,
                content=text_content,
                metadata={
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                },
                page_count=len(reader.pages),
            )
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None
    
    async def process(self, state: ConversationState) -> ConversationState:
        """Process PDF loading and extraction."""
        user_input = state.user_input.lower()
        
        # Try to extract PDF filename from user input
        pdf_filename = None
        if ".pdf" in user_input:
            # Simple extraction - look for .pdf in the input
            words = user_input.split()
            for word in words:
                if word.endswith(".pdf"):
                    pdf_filename = word.strip('"\'')
                    break
        
        pdf_data = None
        if pdf_filename:
            pdf_data = self.load_pdf(pdf_filename)
        
        if pdf_data:
            # Store extracted PDF data
            result = DataCollectionResult(
                agent_type=self.agent_type,
                data=pdf_data.model_dump(),
                success=True,
            )
            state.collected_data[self.agent_type] = result
            
            # Add agent response to history
            state.messages.append({
                "role": "assistant",
                "content": f"PDF loaded successfully: {pdf_data.filename} ({pdf_data.page_count} pages). "
                          f"Extracted {len(pdf_data.content)} characters of text.",
            })
        else:
            # Ask user for PDF file
            state.messages.append({
                "role": "assistant",
                "content": "Please provide the path to the PDF file you'd like me to process. "
                          f"I'll look in {self.pdf_directory} or you can provide an absolute path.",
            })
            
            result = DataCollectionResult(
                agent_type=self.agent_type,
                data={},
                success=False,
                error="PDF file not found or not specified",
            )
            state.collected_data[self.agent_type] = result
        
        return state
    
    def extract_data(self, state: ConversationState) -> DataCollectionResult:
        """Extract structured data."""
        return state.collected_data.get(self.agent_type, DataCollectionResult(
            agent_type=self.agent_type,
            data={},
            success=False,
        ))


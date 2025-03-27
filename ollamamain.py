import os
import time
import logging
import os
import pyautogui
import ollama
import requests
import sqlite3
import glob
from threading import Event
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger('telepathy')
logger.setLevel(logging.DEBUG)

# Configure root logger
# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_handler.setFormatter(formatter)
root_logger.addHandler(root_handler)

handler = logging.StreamHandler()
file_handler = logging.FileHandler('automation.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)

rag_logger = logging.getLogger('rag')
rag_logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

class RagDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('rag.db')
        self._create_tables()

    def _create_tables(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS knowledge
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             context TEXT NOT NULL,
             source TEXT,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS command_history
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             command TEXT NOT NULL,
             success INTEGER,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()

    def add_context(self, context: str, source: str = None):
        self.conn.execute('INSERT INTO knowledge (context, source) VALUES (?, ?)',
                        (context, source))
        self.conn.commit()

    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''SELECT context, source FROM knowledge 
                        WHERE context LIKE ? 
                        ORDER BY timestamp DESC LIMIT ?''',
                     (f'%{query}%', limit))
        return [{'context': row[0], 'source': str(row[1] or '')} for row in cursor.fetchall()]

    def log_command(self, command: str, success: bool):
        self.conn.execute('INSERT INTO command_history (command, success) VALUES (?, ?)',
                        (command, int(success)))
        self.conn.commit()


class ObsidianIndexer:
    def __init__(self, vault_path: str, rag_db: RagDatabase):
        self.vault_path = vault_path
        self.rag_db = rag_db

    def index_notes(self):
        md_files = glob.glob(f'{self.vault_path}/**/*.md', recursive=True)
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.rag_db.add_context(content, source=file_path)
                    logger.info(f'Indexed Obsidian note: {file_path}')
            except Exception as e:
                logger.error(f'Error indexing {file_path}: {str(e)}')


class LearningSystem:
    def __init__(self, rag_db: RagDatabase):
        self.rag_db = rag_db
        self.patterns = {}

    def analyze_usage_patterns(self):
        cursor = self.rag_db.conn.cursor()
        cursor.execute('''SELECT command, COUNT(*) as count 
                        FROM command_history 
                        WHERE success = 1 
                        GROUP BY command 
                        ORDER BY count DESC LIMIT 5''')
        self.patterns = {row[0]: row[1] for row in cursor.fetchall()}
        return self.patterns

    def get_suggested_commands(self) -> List[str]:
        return list(self.patterns.keys())


class AutomationEngine:
    def __init__(self):
        self.running = Event()
        self.screenshot_interval = 1  # Reduced latency interval
        from collections import deque
        self.processed_commands = deque(maxlen=20)  # Track more history to avoid repeats

    def take_screenshot(self):
        screenshot_path = f"temp_ss_{int(time.time())}.png"
        pyautogui.screenshot(screenshot_path)
        logger.info(f"Screenshot captured: {screenshot_path}")
        return screenshot_path

    def get_foreground_app(self):
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            PROCESS_QUERY_INFORMATION = 0x0400
            h_process = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if h_process:
                exe_path = (ctypes.c_char * 512)()
                if ctypes.windll.psapi.GetModuleFileNameExA(h_process, None, ctypes.byref(exe_path), 512):
                    ctypes.windll.kernel32.CloseHandle(h_process)
                    return exe_path.value.decode().split('\\')[-1]
                ctypes.windll.kernel32.CloseHandle(h_process)

            # Fallback to window title
            length = ctypes.windll.user32.GetWindowTextLengthA(hwnd)
            buff = ctypes.create_string_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextA(hwnd, buff, length + 1)
            return buff.value.decode() if buff.value else "Unknown"
        except Exception as e:
            logger.error(f"Failed to get foreground app: {str(e)}")
            return "Unknown"

    def process_image(self, screenshot_path):
        try:
            # Get AI response
            response = ollama.generate(
                model='mistral-nemo',
                prompt=f'''Analyze this active window screenshot and generate precise pyautogui commands using this format:\nCOMMAND: [ACTION] [PARAMETERS]\n\nApplication Context:\n- Foreground App: {self.get_foreground_app()}\n- Screen Resolution: {pyautogui.size()}\n- Recent Commands: {list(self.processed_commands)[-3:]}\n\nCommand Requirements:\n1. CLICK: Provide exact coordinates from visible UI elements\n2. TYPE: Include both text entry and special keys (e.g., {{enter}})\n3. HOTKEY: Use app-specific shortcuts (e.g., alt+f, ctrl+s)\n4. SCROLL: Specify direction and amount\n5. DRAG: Include start/end coordinates\n\nValidation Rules:\n- Coordinates must be within {pyautogui.size()}\n- Type text in quotes, special keys in curly braces\n- Add 'time.sleep(0.5)' between commands\n\nExamples:\nCOMMAND: CLICK 1234,567\nCOMMAND: TYPE "Hello World" {{enter}}\nCOMMAND: HOTKEY ctrl+s\nCOMMAND: SCROLL -200\n\nGenerate 1-3 commands for the current UI state:\n''',
                images=[screenshot_path]
            )
            raw_response = response['response']
            logger.debug(f"Raw AI response:\n{raw_response}")
            return self.parse_commands(raw_response)
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}")
            return []
        finally:
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
                logger.info(f"Deleted screenshot: {screenshot_path}")

    def parse_commands(self, response_text):
        valid_commands = []
        for line in response_text.split('\n'):
            if line.startswith('COMMAND:'):

                cmd_part = line[8:].strip()
                if not cmd_part:
                    continue
                parts = cmd_part.split(' ', 1)
                action = parts[0].upper() if parts else ''
                params = parts[1] if len(parts) > 1 else ''
                if not action:
                    logger.warning(f"Empty command in line: {line}")
                    continue

                # Process command based on action
                if action == 'TYPE':

                    try:
                        text_content = ''
                        special_keys = []
                        if '{' in params and '}' in params:
                            text_part, keys_part = params.split('{', 1)
                            keys_str = keys_part.split('}', 1)[0]
                            special_keys = [k.strip().lower() for k in keys_str.split(',')]
                            text_content = text_part.strip(' \"')
                        else:
                            text_content = params.replace('"', '').strip()

                        if text_content:
                            pyautogui.write(text_content)
                        for key in special_keys:
                            pyautogui.press(key)
                    except Exception as e:
                        logger.error(f"TYPE command failed: {str(e)}")

                    # Handle quoted text with optional special keys
                    import re
                    key_match = re.match(r'^"(.+?)"\s*(\{([^}]+)\})?$', params)
                    if key_match:
                        text_content = key_match.group(1).replace('"', '\\"')
                        special_keys = key_match.group(3)
                        valid_cmd = f'TYPE "{text_content}"'
                        if special_keys:
                            valid_cmd += f' {{{special_keys}}}'
                        valid_commands.append(valid_cmd)
                    else:
                        logger.warning(f"Invalid TYPE command format: {params}")
                if action == 'HOTKEY':
                    normalized_params = '+'.join([p.strip().lower() for p in params.split('+')])
                    command_str = f'HOTKEY {normalized_params}'
                    valid_commands.append(command_str)
                    logger.info(f"Valid command detected: {command_str}")
                elif action == 'CLICK':
                    if ',' in params:
                        x, y = params.split(',', 1)
                        if x.strip().isdigit() and y.strip().isdigit():
                            screen_width, screen_height = pyautogui.size()
                            if 0 <= int(x) <= screen_width and 0 <= int(y) <= screen_height:
                                valid_commands.append(f'CLICK {x.strip()},{y.strip()}')
                                logger.info(f"Valid click command: {params}")
                            else:
                                logger.warning(f"Coordinates out of bounds: {params}")
                        else:
                            logger.warning(f"Invalid CLICK coordinates: {params}")
                else:
                    logger.warning(f"Invalid command ignored: {cmd_part}")
                continue
        
        logger.info(f"Valid commands identified: {len(valid_commands)}")
        # Execute all valid new commands
        new_commands = [cmd for cmd in valid_commands if cmd not in self.processed_commands]
        logger.info(f"New commands to execute: {new_commands}")
        
        logger.debug(f"Command filtering - New: {len(new_commands)}, Already processed: {len(valid_commands)-len(new_commands)}")
        logger.debug(f"All valid commands: {valid_commands}")
        logger.debug(f"Processed commands: {list(self.processed_commands)}")
        
        return new_commands

    def execute_command(self, command):
        try:
            logger.debug(f"Executing command: {command}")
            parts = command.split(' ', 1)
            action = parts[0]
            params = parts[1] if len(parts) > 1 else ''

            if action == 'CLICK':
                if params.count(',') == 1:
                    x, y = map(int, params.split(','))
                else:
                    logger.error(f"Invalid CLICK coordinates: {params}")
                    return
                logger.debug(f"Click coordinates: X={x}, Y={y}")
                pyautogui.moveTo(x, y, duration=0.25)
                pyautogui.click(x, y)
                logger.info(f"Clicked at ({x},{y})")
            elif action == 'TYPE':
                try:
                    text_content = ''
                    special_keys = []
                    if '{' in params and '}' in params:
                        text_part, keys_part = params.split('{', 1)
                        keys_str = keys_part.split('}', 1)[0]
                        special_keys = [k.strip().lower() for k in keys_str.split(',')]
                        text_content = text_part.strip(' \"')
                    else:
                        text_content = params.replace('"', '').strip()

                    if text_content:
                        pyautogui.write(text_content)
                    for key in special_keys:
                        pyautogui.press(key)
                except Exception as e:
                    logger.error(f"TYPE command failed: {str(e)}")
                # Already handled in try block above
            elif action == 'HOTKEY':
                keys = params.split('+')
                logger.debug(f"Hotkey components: {keys}")
                pyautogui.hotkey(*keys)
                logger.info(f"Pressed: {params}")

            self.processed_commands.append(command)
        except pyautogui.FailSafeException:
            logger.error("Fail-safe triggered! Move mouse to corner to disable.")
        except ValueError as e:
            logger.error(f"Invalid command format: {str(e)}")
        except Exception as e:
            logger.error(f"Command failed '{command}': {str(e)}", exc_info=True)

    def run_loop(self):
        self.running.set()
        logger.info("Starting AI-Automation loop")
        while self.running.is_set():
            try:
                ss_path = self.take_screenshot()
                commands = self.process_image(ss_path)
                for cmd in commands:
                    self.execute_command(cmd)
                time.sleep(self.screenshot_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {str(e)}")
                time.sleep(1)

    def stop(self):
        self.running.clear()
        logger.info("Automation loop stopped")

if __name__ == '__main__':
    engine = AutomationEngine()
    try:
        engine.run_loop()
    except KeyboardInterrupt:
        engine.stop()
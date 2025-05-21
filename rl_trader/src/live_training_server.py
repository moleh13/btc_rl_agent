# /Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/live_training_server.py

import asyncio
import websockets
import threading
import json
import logging
import time

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiveTrainingServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.server_thread = None
        self.server_instance = None # Will store the actual server object from websockets.serve
        self._stop_event = threading.Event()

    async def _client_handler(self, websocket):
        logging.info(f"Client connected from {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            async for message in websocket: # Keep connection open
                # logging.debug(f"Received message (ignored): {message}")
                pass
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            logging.info(f"Client {websocket.remote_address} disconnected.")
        except Exception as e:
            logging.error(f"Error with client {websocket.remote_address}: {e}")
        finally:
            # logging.info(f"Removing client {websocket.remote_address}") # Can be noisy
            self.clients.remove(websocket)

    async def _broadcast(self, data_dict):
        if not self.clients:
            return

        message = json.dumps(data_dict)
        # Use a copy of self.clients in case it's modified during iteration
        # (though less likely with asyncio.gather if client_handler is robust)
        current_clients = list(self.clients) 
        if not current_clients:
            return

        tasks = [client.send(message) for client in current_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client_to_remove = current_clients[i]
                logging.warning(f"Failed to send to {client_to_remove.remote_address}: {result}. Removing.")
                # Attempt to remove directly, handler should also catch this
                if client_to_remove in self.clients:
                    self.clients.remove(client_to_remove)
                try: # Attempt to close from server-side if send failed
                    await client_to_remove.close()
                except: pass


    def broadcast_data(self, data_dict):
        if self.loop and self.loop.is_running() and not self._stop_event.is_set():
            future = asyncio.run_coroutine_threadsafe(self._broadcast(data_dict), self.loop)
            try:
                # future.result(timeout=0.1) # Optional: slight block for ensuring send starts
                pass
            except TimeoutError:
                logging.warning("Broadcast data send schedule timed out.")
            except Exception as e:
                logging.error(f"Error scheduling broadcast: {e}")
        else:
            logging.debug("Server loop not running or stop event set, not broadcasting.")

    async def _main_server_logic(self):
        """The main coroutine that runs the server and waits for stop signal."""
        try:
            # websockets.serve returns a Server object
            # Explicitly pass the bound method
            handler_coroutine_factory = self._client_handler 
            self.server_instance = await websockets.serve(
                handler_coroutine_factory, self.host, self.port # loop=self.loop is often implicit now
            )
            logging.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            # Keep server running until stop event is set
            await self._stop_event.wait() # Async wait for the threading.Event
        except OSError as e:
             logging.error(f"Could not start WebSocket server (OSError): {e}")
             # Signal main thread that server didn't start properly
             # This requires a way to communicate back, or check self.server_instance after start
        except Exception as e:
            logging.error(f"Exception in main server logic: {e}")
        finally:
            if self.server_instance:
                self.server_instance.close()
                await self.server_instance.wait_closed()
                logging.info("WebSocket server instance gracefully closed.")

    def _run_server_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logging.info(f"Asyncio event loop created for server thread: {self.loop}")

        # Convert threading.Event to an asyncio.Event for use in async code if needed
        # Or, more simply, make _main_server_logic itself check the threading.Event periodically
        # For this version, let's make an asyncio equivalent for cleaner async waiting
        async_stop_event = asyncio.Event() # This is an asyncio.Event
        
        # Create a wrapper for the threading.Event to set the asyncio.Event
        def monitor_stop_event():
            self._stop_event.wait() # Blocks this small helper until threading.Event is set
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(async_stop_event.set)

        # Start a small helper thread to monitor the threading.Event
        # This is one way to bridge threading.Event to asyncio.Event
        stop_monitor_thread = threading.Thread(target=monitor_stop_event, daemon=True)
        stop_monitor_thread.start()

        async def main_server_with_async_stop():
            try:
                self.server_instance = await websockets.serve(
                    self._client_handler, self.host, self.port
                )
                logging.info(f"WebSocket server started on ws://{self.host}:{self.port} (instance: {self.server_instance})")
                await async_stop_event.wait() # Wait for the asyncio event
            except OSError as e:
                logging.error(f"Could not start WebSocket server (OSError in main_server_with_async_stop): {e}")
                self._stop_event.set() # Also set the threading event to signal failure
            except Exception as e:
                logging.error(f"Exception in main_server_with_async_stop: {e}")
            finally:
                if self.server_instance:
                    self.server_instance.close()
                    # Ensure wait_closed is awaited to prevent warnings
                    if hasattr(self.server_instance, 'wait_closed'):
                         await self.server_instance.wait_closed()
                    logging.info("WebSocket server instance gracefully closed (from main_server_with_async_stop).")
        
        try:
            self.loop.run_until_complete(main_server_with_async_stop())
        finally:
            # Ensure the loop is properly cleaned up
            # Cancel all remaining tasks
            if hasattr(asyncio, 'all_tasks'): # Python 3.7+
                tasks = asyncio.all_tasks(loop=self.loop)
            else:
                tasks = asyncio.Task.all_tasks(loop=self.loop) # Older Python

            for task in tasks:
                task.cancel()
            if tasks:
                self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            self.loop.close()
            logging.info("Asyncio event loop closed in _run_server_loop.")
            if not self._stop_event.is_set(): # If loop exited for other reasons
                 self._stop_event.set()


    def start(self):
        if self.server_thread and self.server_thread.is_alive():
            logging.warning("Server thread is already running.")
            return True # Or False if this is an error state

        self._stop_event.clear()
        self.server_thread = threading.Thread(target=self._run_server_loop, daemon=True)
        self.server_thread.start()
        logging.info("WebSocket server thread starting...")
        
        # Give the server some time to start and check if it failed
        time.sleep(1.0) # Increased sleep for robustness
        if self._stop_event.is_set() and not self.server_instance:
            logging.error("Server failed to start (stop_event set and no server_instance). Check logs for OSError.")
            return False
        if not self.server_instance:
            logging.warning("Server instance not confirmed after startup delay. May not be listening.")
            # This might still be okay if it's just slow, but it's a warning sign.
            # For critical startup, a more robust check (e.g., trying to connect) would be needed.
            return False # Consider it a failure if instance isn't up
        
        logging.info(f"Server thread started. Server instance: {self.server_instance}")
        return True

    def stop(self):
        if not self.server_thread or not self.server_thread.is_alive():
            # logging.info("Server thread is not running or already stopped.")
            return

        logging.info("Stopping WebSocket server...")
        self._stop_event.set() # Signal the server loop to stop

        # The asyncio event loop and its tasks should handle shutdown now.
        # Joining the thread waits for it to complete.
        self.server_thread.join(timeout=5)
        if self.server_thread.is_alive():
            logging.warning("Server thread did not stop in time.")
        else:
            logging.info("Server thread stopped.")
        
        self.server_thread = None
        self.loop = None
        self.server_instance = None # Clear the instance


if __name__ == '__main__':
    server = LiveTrainingServer(host="localhost", port=8765)
    print("Starting server from main thread...")
    if not server.start():
        print("Failed to start server. Exiting.")
        exit()
    
    print("Server started. Main thread will now send test messages for 10 seconds.")
    try:
        for i in range(100): # Send 100 messages over 10 seconds
            if not (server.server_thread and server.server_thread.is_alive()) or server._stop_event.is_set():
                print("Server not running or stop event set. Exiting test loop.")
                break
            test_data = {
                "type": "step_data", "step": i, "price": 10000 + i * 10,
                "equity": 50000 + i * 50, "action": (i % 20 - 10) / 10.0
            }
            server.broadcast_data(test_data)
            # print(f"Sent test message {i}") # Can be too verbose
            time.sleep(0.1)
        
        if server.server_thread and server.server_thread.is_alive() and not server._stop_event.is_set():
            server.broadcast_data({"type": "episode_reset", "episode_number": 2})
            print("Sent episode reset message.")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        print("Stopping server from main thread...")
        server.stop()
        print("Server stopped. Exiting main.")
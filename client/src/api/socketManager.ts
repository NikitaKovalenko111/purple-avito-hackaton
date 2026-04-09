let currentSocket: WebSocket | null = null;

export const socketService = {
    connect: (url: string, onMessage: (ev: MessageEvent) => void) => {
        if (currentSocket) {
            currentSocket.close();
        }

        currentSocket = new WebSocket(url);
        
        currentSocket.onmessage = onMessage;
        currentSocket.onopen = () => console.log('WS connected');
        currentSocket.onclose = () => { currentSocket = null; };
        
        return currentSocket;
    },

    disconnect: () => {
        currentSocket?.close();
        currentSocket = null;
    }
};
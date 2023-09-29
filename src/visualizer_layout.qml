import QtQuick 2.7
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    //focus: true
    //title of the application
    title: qsTr("Hello World")
    id: appWindow

    //property string imageSVG: "file:///data/uni/TFG-python/empty_board.svg"
    property string imageSVG: backend.refreshSVG() 
    
    //property int maxCluster: ""

    width: 1000
    height: 800
    
    Shortcut {
        sequence: "Left"
        onActivated: {
            positionText.text = (parseInt(positionText.text)-1).toString();
            svgBoard.reload();
        }
    }
    Shortcut {
        sequence: "Right"
        onActivated: {
            positionText.text = (parseInt(positionText.text)+1).toString();
            svgBoard.reload();
        }
    }
    Shortcut {
        sequence: "Up"
        onActivated: {
            clusterText.text = (parseInt(clusterText.text)+1).toString(); 
            svgBoard.reload();
        }
    }
    Shortcut {
        sequence: "Down"
        onActivated: {
            clusterText.text = (parseInt(clusterText.text)-1).toString(); 
            svgBoard.reload();
        }
    }

    header:ToolBar {
        id: stuff
        RowLayout {
            //anchors.fill: parent
            Item{Layout.preferredWidth: 20}
            Text {
                text: qsTr("Cluster:")
            }
            TextField {
                id: clusterText
                text: qsTr("0")
                Layout.preferredWidth: 60
                validator: IntValidator{bottom: 0;top: backend.maxCluster}
                horizontalAlignment: TextInput.AlignHCenter
            }
            Button {
                text: qsTr("Go to cluster")
                background: Rectangle {
                color: parent.down ? "#bbbbbb" :
                        (parent.hovered ? "#d6d6d6" : "#aaaaaa")}
                onClicked: {
                    svgBoard.reload();
                }
            }
            Item{Layout.preferredWidth: 100}
            Text {
                text: qsTr("Position:")
            }
            TextField {
                id: positionText
                text: qsTr("0")
                Layout.preferredWidth: 60
                validator: IntValidator{bottom: 0;top: backend.maxPosition}
                horizontalAlignment: TextInput.AlignHCenter
            }
            Button {
                text: qsTr("Go to position")
                background: Rectangle {
                color: parent.down ? "#bbbbbb" :
                        (parent.hovered ? "#d6d6d6" : "#aaaaaa")}
                onClicked: {
                    svgBoard.reload();
                }
            }
            Button {
                text: qsTr("Random position")
                background: Rectangle {
                color: parent.down ? "#bbbbbb" :
                        (parent.hovered ? "#d6d6d6" : "#aaaaaa")}
            }
            Item{Layout.preferredWidth: 20}
                
            
        }
    } // end of header

    Button {
        text: qsTr("Save Image")
        background: Rectangle {
        color: parent.down ? "#bbbbbb" :
                (parent.hovered ? "#d6d6d6" : "#aaaaaa")}
        x:664
        y:32
    }
    Text {
        text: "Positions of the cluster:"+backend.maxPosition
        x:664
        y:100
    }
    Text {
        text: "Number of clusters:"+backend.maxCluster
        x:664
        y:200
    }
        
    Image {
        id: svgBoard
        focus: true
        x:32
        y:32
        width: 600
        height: 600
        source: imageSVG 
        sourceSize: Qt.size(width, height)
        function reload() {
            var tmp = parseInt(clusterText.text);
            if (tmp >= backend.maxCluster){
                tmp = 0;
                clusterText.text = tmp.toString();
            }
            else if (tmp < 0) {
                tmp = backend.maxCluster-1 
                clusterText.text = tmp.toString();
            }
            backend.setCluster(tmp);

            tmp = parseInt(positionText.text)
            if (tmp >= backend.maxPosition){
                tmp = 0;
                positionText.text = tmp.toString();
            }
            else if (tmp < 0) {
                tmp = backend.maxPosition-1;
                positionText.text = tmp.toString();
            }
            backend.setPosition(tmp);

            imageSVG = backend.refreshSVG();

            //source = "";
            source = imageSVG;
        }
    }

}

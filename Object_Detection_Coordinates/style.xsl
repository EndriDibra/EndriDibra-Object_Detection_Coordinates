<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <style>
          table {
            border-collapse: collapse;
            width: 100%;
          }
          th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
          }
          th {
            background-color: #f2f2f2;
          }
        </style>
      </head>
      <body>
        <h1>X, Y Coordinates</h1>
        <table>
          <thead>
            <tr>
              <th>X Coordinate</th>
              <th>Y Coordinate</th>
            </tr>
          </thead>
          <tbody>
            <xsl:for-each select="coordinates/point">
              <tr>
                <td><xsl:value-of select="x"/></td>
                <td><xsl:value-of select="y"/></td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>

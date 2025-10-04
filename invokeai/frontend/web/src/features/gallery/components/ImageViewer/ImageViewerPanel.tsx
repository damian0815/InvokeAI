import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare, selectImageToTokenize, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';

import { ImageViewerContextProvider } from './context';
import { ImageComparison } from './ImageComparison';
import { ImageViewer } from './ImageViewer';
import { VideoViewer } from './VideoViewer';
import { ImageTokenization } from './ImageTokenization';

const selectIsComparing = createSelector(
  [selectLastSelectedItem, selectImageToCompare],
  (lastSelectedImage, imageToCompare) => !!lastSelectedImage && !!imageToCompare
);

const selectIsTokenizing = createSelector([selectImageToTokenize], 
  (imageToTokenize) => !!imageToTokenize
);

export const ImageViewerPanel = memo(() => {
  const isComparing = useAppSelector(selectIsComparing);
  const isTokenizing = useAppSelector(selectIsTokenizing);
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);

  const notComparingOrTokenizing = !isComparing && !isTokenizing;

  return (
    <ImageViewerContextProvider>
      {
        // The image viewer renders progress images - if no image is selected, show the image viewer anyway
        notComparingOrTokenizing && !lastSelectedItem && <ImageViewer />
      }
      {notComparingOrTokenizing && lastSelectedItem?.type === 'image' && <ImageViewer />}
      {notComparingOrTokenizing && lastSelectedItem?.type === 'video' && <VideoViewer />}
      {isComparing && <ImageComparison />}
      {isTokenizing && <ImageTokenization />}
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';
